"""Listing 1.9 — Bird's-eye-view bounding-box IoU for 3D detection evaluation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Box2D:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    def validate(self):
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("invalid box extents")

def bev_iou(box_a,box_b):
    a=Box2D(*box_a); b=Box2D(*box_b); a.validate(); b.validate()
    inter_x_min=max(a.x_min,b.x_min); inter_y_min=max(a.y_min,b.y_min)
    inter_x_max=min(a.x_max,b.x_max); inter_y_max=min(a.y_max,b.y_max)
    inter_w=max(0.0,inter_x_max-inter_x_min)
    inter_h=max(0.0,inter_y_max-inter_y_min)
    inter_area=inter_w*inter_h
    area_a=(a.x_max-a.x_min)*(a.y_max-a.y_min)
    area_b=(b.x_max-b.x_min)*(b.y_max-b.y_min)
    union=area_a+area_b-inter_area
    return inter_area/union if union>0 else 0.0

if __name__=="__main__":
    print("IoU:",bev_iou((0,0,2,2),(1,1,3,3)))
