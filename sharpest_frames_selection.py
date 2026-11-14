"""Decoder-independent, static-mask-aware synchronized video selection.

Original CLI behavior is unchanged. The caller supplies explicitly synchronized
native/perspective views and image-bound static masks after any calibrated crop
and alignment. This module never decodes a video or chooses one lens separately.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import math
import statistics
from itertools import islice

import numpy as np

VERSION = 'masked-synchronized-laplacian-v1'


def masked_laplacian_variance(gray, valid, *, minimum_pixels=1024):
    """Variance of the 4-neighbour Laplacian on eroded valid STATIC pixels.

    Eroding the mask footprint prevents excluded people, black projection rims
    and their artificial edges from increasing sharpness. All candidates must
    use the same declared raster/scoring scale; scores are not resolution-free.
    Returns a score and the number of contributing pixels.
    """
    image=np.asarray(gray);mask=np.asarray(valid)
    if image.ndim!=2 or mask.shape!=image.shape or min(image.shape)<3:
        raise ValueError('Provide matching two-dimensional grayscale and static-mask rasters')
    if image.dtype.kind not in 'uif' or not np.isfinite(image).all():
        raise ValueError('Sharpness requires finite numeric grayscale pixels')
    if mask.dtype==np.bool_:
        keep=mask
    elif mask.dtype.kind in 'ui' and np.isin(mask,[0,1,255]).all():
        keep=mask>0
    else:
        raise ValueError('Static validity must be boolean or a binary integer mask')
    if type(minimum_pixels) is not int or minimum_pixels<1:
        raise ValueError('Minimum static support must be a positive integer')
    supported=(keep[1:-1,1:-1]&keep[:-2,1:-1]&keep[2:,1:-1]&keep[1:-1,:-2]&keep[1:-1,2:])
    count=int(supported.sum())
    if count<minimum_pixels:
        return 0.,count
    image=image.astype(np.float64,copy=False)
    lap=(image[:-2,1:-1]+image[2:,1:-1]+image[1:-1,:-2]+image[1:-1,2:]-4*image[1:-1,1:-1])
    return float(lap[supported].var()),count


@dataclass
class SynchronizedCandidate:
    window_id: str
    capture_id: str
    pts: int
    time_base: tuple[int,int]
    frame_index: int
    scores: dict[str,float]
    static_pixels: dict[str,int]
    provenance: dict = field(default_factory=dict)

    @property
    def time(self):
        return self.pts*Fraction(*self.time_base)

    def validate(self):
        if (not isinstance(self.window_id,str) or not self.window_id
                or not isinstance(self.capture_id,str) or not self.capture_id
                or type(self.pts) is not int or type(self.frame_index) is not int or self.frame_index<0
                or not isinstance(self.time_base,(tuple,list)) or len(self.time_base)!=2 or any(type(v) is not int or v<=0 for v in self.time_base)):
            raise ValueError('Each candidate needs an explicit window, shared capture identity and rational PTS')
        if not self.scores or set(self.scores)!=set(self.static_pixels):
            raise ValueError('Every synchronized sensor needs a score and static-support count')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in self.scores.values()):
            raise ValueError('Sharpness scores must be finite and nonnegative')
        if any(type(v) is not int or v<0 for v in self.static_pixels.values()):
            raise ValueError('Static-support counts must be nonnegative integers')


def select_synchronized(candidates, *, relative_floor=.4, absolute_floor=0., window_half=5,
                        minimum_pixels=1024, maximum_candidates_per_window=3):
    """Select ONE shared PTS per time window, then apply per-lens focus checks.

    Rank by the weaker lens's score relative to its window median; tie-break by
    geometric mean and source frame index. A sharp textured lens cannot hide a
    blurred second lens. Return complete kept/rejected receipts, never silently
    fill a rejected window or claim camera coverage/overlap has been verified.
    """
    candidates=list(islice(iter(candidates),100_001))
    if not candidates or len(candidates)>100_000:
        raise ValueError('Supply a bounded nonempty synchronized candidate sequence')
    if (not 0<=relative_floor<=1 or not math.isfinite(absolute_floor) or absolute_floor<0
            or type(minimum_pixels) is not int or minimum_pixels<1
            or type(window_half) is not int or not 0<=window_half<=100
            or type(maximum_candidates_per_window) is not int or not 1<=maximum_candidates_per_window<=100):
        raise ValueError('Invalid synchronized sharp-frame selection settings')
    groups={};sensors=None;identities=set()
    for candidate in candidates:
        candidate.validate();keys=set(candidate.scores)
        if sensors is None:sensors=keys
        if keys!=sensors:raise ValueError('All candidates must contain the same complete synchronized sensor set')
        if candidate.capture_id in identities:raise ValueError('Candidate capture identities must be unique')
        identities.add(candidate.capture_id);groups.setdefault(candidate.window_id,[]).append(candidate)
    chosen=[]
    for window,rows in groups.items():
        if len(rows)>maximum_candidates_per_window:raise ValueError('Candidate time window exceeds its bounded decode budget')
        baseline={s:statistics.median(r.scores[s] for r in rows) for s in sensors}
        def rank(row):
            normalized=[row.scores[s]/baseline[s] if baseline[s]>0 else 0. for s in sensors]
            eligible=all(row.static_pixels[s]>=minimum_pixels for s in sensors)
            joint=math.exp(sum(math.log(max(v,1e-30)) for v in normalized)/len(normalized))
            return eligible,min(normalized),joint,-row.frame_index
        best=max(rows,key=rank)
        chosen.append({'candidate':best,'candidates':rows,'joint_min_normalized_score':rank(best)[1]})
    chosen.sort(key=lambda row:(row['candidate'].time,row['candidate'].window_id))
    result=[]
    for index,item in enumerate(chosen):
        candidate=item['candidate'];medians={};reasons=[]
        neighbors=chosen[max(0,index-window_half):index]+chosen[index+1:index+window_half+1]
        for sensor in sorted(sensors):
            median=statistics.median(r['candidate'].scores[sensor] for r in neighbors) if neighbors else candidate.scores[sensor]
            medians[sensor]=median
            if candidate.static_pixels[sensor]<minimum_pixels:reasons.append(sensor+': insufficient unmasked static pixels')
            if candidate.scores[sensor]<absolute_floor:reasons.append(sensor+': below absolute sharpness floor')
            if median>0 and candidate.scores[sensor]<relative_floor*median:reasons.append(sensor+': below local relative sharpness floor')
        result.append({'window_id':candidate.window_id,'capture_id':candidate.capture_id,'pts':candidate.pts,
            'time_base':list(candidate.time_base),'time_seconds':float(candidate.time),'frame_index':candidate.frame_index,
            'scores':dict(candidate.scores),'static_pixels':dict(candidate.static_pixels),'local_medians':medians,
            'joint_min_normalized_score':item['joint_min_normalized_score'],'kept':not reasons,'reasons':reasons,
            'provenance':candidate.provenance,'candidate_scores':[{'capture_id':r.capture_id,'pts':r.pts,
                'time_base':list(r.time_base),'frame_index':r.frame_index,'scores':dict(r.scores),
                'static_pixels':dict(r.static_pixels),'selected':r is candidate} for r in item['candidates']]})
    return {'schema_version':1,'version':VERSION,'sensors':sorted(sensors),'windows':result,
        'kept_windows':sum(row['kept'] for row in result),'total_windows':len(result),
        'settings':{'relative_floor':relative_floor,'absolute_floor':absolute_floor,'window_half':window_half,
                    'minimum_pixels':minimum_pixels,'maximum_candidates_per_window':maximum_candidates_per_window},
        'coverage_verified':False,'camera_overlap_verified':False,
        'policy':'shared PTS per physical rig; static masks required; rejected windows remain explicit'}
