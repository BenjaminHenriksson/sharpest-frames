"""Field-wise DJI SRT telemetry and lossless PNG EXIF/XMP metadata.

Missing optional fields remain missing. SRT wall-clock timestamps carry no
invented timezone, gimbal angles, physical focal length or camera calibration.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

SRT_VERSION = 'dji-fieldwise-srt-png-v1'

def _rational(value):
    numerator, slash, denominator = str(value).partition('/')
    return Fraction(numerator)/Fraction(denominator) if slash else Fraction(numerator)



def _milliseconds(value):
    hours, minutes, rest = value.replace(',', '.').split(':')
    return int(hours)*3600000+int(minutes)*60000+round(Fraction(rest)*1000)


def parse_dji_srt(text):
    """Return zero-based source frame indices with original available fields.

    Accept DJI clips without gimbal metadata (e.g. Mini 4 Pro) as well as the
    more complete M3E fields. Reject duplicate identities and invalid intervals.
    """
    if not isinstance(text, str) or len(text.encode('utf8')) > 64*1024**2:
        raise ValueError('Provide bounded UTF-8 DJI SRT telemetry')
    result = {}
    numeric = {'iso','shutter','fnum','ev','focal_len','latitude','longitude',
               'rel_alt','abs_alt','gb_yaw','gb_pitch','gb_roll','ct'}
    for block in re.split(r'\r?\n\s*\r?\n', text):
        index = re.search(r'FrameCnt:\s*(\d+)', block)
        interval = re.search(r'(\d{2}:\d{2}:\d{2}[,.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d+)', block)
        if not index or not interval:
            continue
        frame = int(index.group(1))-1
        start,end = (_milliseconds(value) for value in interval.groups())
        if frame < 0 or frame in result or end <= start:
            raise ValueError('DJI SRT contains duplicate frames or invalid time intervals')
        row = {'frame_index': frame, 'start_milliseconds': start, 'end_milliseconds': end}
        wall = re.search(r'\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)', block)
        if wall:
            row['datetime'] = wall.group(1)
        for key,value in re.findall(r'\b([a-z_]+):\s*([^\]\s<>]+)', block):
            if key not in numeric:
                continue
            try:
                number = _rational(value) if key == 'shutter' else float(value)
            except (ValueError,ZeroDivisionError):
                raise ValueError('DJI SRT contains an invalid numeric telemetry field') from None
            if not math.isfinite(float(number)):
                raise ValueError('DJI SRT contains nonfinite telemetry')
            row[key] = value
        if 'latitude' in row and not -90 <= float(row['latitude']) <= 90:
            raise ValueError('DJI SRT latitude is outside its valid range')
        if 'longitude' in row and not -180 <= float(row['longitude']) <= 180:
            raise ValueError('DJI SRT longitude is outside its valid range')
        colour = re.search(r'\[color_md:\s*([^\]]+)\]', block)
        if colour: row['color_md'] = colour.group(1).strip()
        result[frame] = row
    return result


def matched_telemetry(rows, frame_index, pts, time_base):
    """Bind exact decoded PTS to the same SRT FrameCnt; never nearest-guess."""
    row = rows.get(frame_index)
    if row is None:
        return None
    milliseconds = Fraction(pts*time_base[0]*1000, time_base[1])
    # A half-millisecond tolerance accounts for SRT's printed millisecond grid.
    if not Fraction(row['start_milliseconds'])-Fraction(1,2) <= milliseconds < Fraction(row['end_milliseconds'])+Fraction(1,2):
        raise ValueError('Selected frame presentation time does not match its SRT interval')
    return dict(row)


def png_metadata(meta, *, camera_make=None, camera_model=None):
    """Return Pillow EXIF and PNG iTXt metadata for known source values only.

    DJI SRT focal_len is retained verbatim in XMP. It is not silently labelled
    as physical millimetres; camera-specific equivalence needs its own authority.
    """
    from PIL import Image, PngImagePlugin, TiffImagePlugin
    rational = TiffImagePlugin.IFDRational
    exif = Image.Exif(); photo = {}
    if camera_make:
        exif[271] = str(camera_make)
    if camera_model:
        exif[272] = str(camera_model)
    exif[274] = 1
    fields = []
    if meta:
        if meta.get('datetime'):
            main,_,sub = meta['datetime'].partition('.')
            main = main.replace('-', ':', 2)
            photo[36867] = main; photo[36868] = main; exif[306] = main
            if sub:
                photo[37521] = sub; photo[37522] = sub
            fields.append('datetime')
        for key,tag in [('iso',34855),('fnum',33437),('shutter',33434),('ev',37380)]:
            if key in meta:
                number = _rational(meta[key])
                if key != 'ev' and number <= 0:
                    continue
                photo[tag] = int(number) if key == 'iso' else rational(number.numerator,number.denominator)
                fields.append(key)
        gps = {}
        def dms(value):
            number = abs(Fraction(value));degrees = int(number);minutes = int((number-degrees)*60)
            seconds = ((number-degrees)*60-minutes)*60
            return rational(degrees),rational(minutes),rational(seconds.numerator,seconds.denominator)
        if 'latitude' in meta and 'longitude' in meta:
            gps.update({0:bytes([2,3,0,0]),1:'N' if float(meta['latitude'])>=0 else 'S',2:dms(meta['latitude']),
                        3:'E' if float(meta['longitude'])>=0 else 'W',4:dms(meta['longitude'])})
            fields.extend(['latitude','longitude'])
        if 'abs_alt' in meta:
            altitude = Fraction(meta['abs_alt']);gps[5] = 0 if altitude >= 0 else 1
            altitude = abs(altitude);gps[6] = rational(altitude.numerator,altitude.denominator)
            fields.append('abs_alt')
        if gps:
            exif[34853] = gps
    if photo: exif[34665] = photo
    values = {'MetadataVersion':SRT_VERSION,'TimestampTimezone':'unspecified by source SRT',
              'CameraCalibration':'not supplied','FocalLengthInterpretation':'original SRT value; not asserted physical focal length'}
    if meta:
        values.update({'SourceFrameIndex':str(meta['frame_index']),'SourceStartMilliseconds':str(meta['start_milliseconds']),
                       'SourceEndMilliseconds':str(meta['end_milliseconds'])})
        for key,value in meta.items():
            if key not in {'frame_index','start_milliseconds','end_milliseconds'}:
                values['SRT_'+key] = str(value)
        fields.extend(key for key in meta if key not in fields)
    attrs = ' '.join('gi:'+key+'="'+escape(value,{'"':'&quot;'})+'"' for key,value in values.items())
    dji = {}
    for source,target in [('abs_alt','AbsoluteAltitude'),('rel_alt','RelativeAltitude'),('gb_yaw','GimbalYawDegree'),
                          ('gb_pitch','GimbalPitchDegree'),('gb_roll','GimbalRollDegree')]:
        if meta and source in meta:
            dji[target] = meta[source]
    attrs += ''.join(' drone-dji:'+key+'="'+escape(value,{'"':'&quot;'})+'"' for key,value in dji.items())
    xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           '<rdf:Description rdf:about="" xmlns:gi="https://groundinference.com/ns/video/1.0/" '
           'xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '+attrs+'/></rdf:RDF></x:xmpmeta>')
    info = PngImagePlugin.PngInfo(); info.add_itxt('XML:com.adobe.xmp', xmp)
    return exif,info,sorted(set(fields))


def save_png_with_telemetry(image, path, meta, *, camera_make=None, camera_model=None):
    """Encode lossless PNG once, including metadata before content hashing."""
    exif,info,fields = png_metadata(meta,camera_make=camera_make,camera_model=camera_model)
    image.save(Path(path),format='PNG',compress_level=9,exif=exif,pnginfo=info)
    # This is publication integrity for the actual requested extraction, not a
    # separate test/probe: conventional SfM readers require the Photo sub-IFD.
    from PIL import Image
    with Image.open(path) as saved:
        if saved.size != image.size or saved.mode != image.mode:
            raise ValueError('Lossless frame raster identity changed while saving')
        stored = saved.getexif(); photo = stored.get_ifd(34665)
        if stored.get(274) != 1:
            raise ValueError('Saved frame orientation metadata is missing')
        required = {key:tag for key,tag in [('iso',34855),('fnum',33437),('shutter',33434),('ev',37380)]
                    if meta and key in meta}
        for key,tag in required.items():
            if tag not in photo or abs(float(photo[tag])-float(_rational(meta[key]))) > 1e-6:
                raise ValueError('Saved frame EXIF exposure metadata did not match the source')
        if meta and meta.get('datetime'):
            main,_,sub = meta['datetime'].partition('.')
            if photo.get(36867) != main.replace('-',':',2) or (sub and photo.get(37521) != sub):
                raise ValueError('Saved frame EXIF source timestamp is missing or changed')
        if meta and 'latitude' in meta and 'longitude' in meta:
            gps = stored.get_ifd(34853)
            if any(tag not in gps for tag in (1,2,3,4)):
                raise ValueError('Saved frame EXIF source GPS is missing')
            def decimal(values):
                return float(values[0])+float(values[1])/60+float(values[2])/3600
            lat = decimal(gps[2])*(-1 if gps[1] == 'S' else 1)
            lon = decimal(gps[4])*(-1 if gps[3] == 'W' else 1)
            if abs(lat-float(meta['latitude']))>1e-7 or abs(lon-float(meta['longitude']))>1e-7:
                raise ValueError('Saved frame EXIF GPS differs from its SRT source')
        xmp = saved.info.get('XML:com.adobe.xmp','')
        if SRT_VERSION not in xmp or (meta and 'SourceFrameIndex="'+str(meta['frame_index'])+'"' not in xmp):
            raise ValueError('Saved frame XMP source binding is missing')
    return fields
