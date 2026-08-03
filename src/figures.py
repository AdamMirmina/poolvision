"""Figures for the writeup, drawn from the measurements themselves."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ASSETS = Path("C:/dev/poolean/web/public/assets")

def main():
    import cv2, numpy as np
    BG=(18,20,26); INK=(238,240,245); DIM=(150,158,172)
    GOOD=(128,222,74); WARN=(96,180,247); BAD=(90,90,235); LINE=(58,64,78)
    def txt(im,s,p,sc=0.6,c=INK,th=2): cv2.putText(im,s,p,cv2.FONT_HERSHEY_SIMPLEX,sc,c,th,cv2.LINE_AA)

    # Which signals actually earn their place, and which were measured and dropped.
    im=np.full((760,1400,3),BG,np.uint8)
    txt(im,"Every rule tried, and whether it survived measurement",(48,62),0.85,INK,2)
    txt(im,"Percentages are how often the rule is right when it fires. The bar to beat is the base rate.",(48,100),0.52,DIM,1)
    rows=[("Drop zone: ball never fell under the net",94,69,GOOD,"kept"),
          ("Net shimmer: net never moved",92,74,GOOD,"kept"),
          ("Facing left/right from the wrists",89,None,GOOD,"kept"),
          ("Facing front/back from the face",81,None,WARN,"replaced"),
          ("Ball's top inside the rim ellipse",38,None,BAD,"too rare to use"),
          ("Ball ever overlaps the rim",67,69,BAD,"below chance"),
          ]
    y0=160
    for i,(lab,v,base,col,note) in enumerate(rows):
        y=y0+i*92
        cv2.rectangle(im,(48,y),(1180,y+52),(30,34,42),-1)
        cv2.rectangle(im,(48,y),(48+int(1132*v/100),y+52),col,-1)
        if base:
            bx=48+int(1132*base/100)
            cv2.line(im,(bx,y-6),(bx,y+58),(255,255,255),2)
        txt(im,lab,(60,y+34),0.55,(16,18,24) if v>60 else INK,2)
        txt(im,f"{v}%",(1196,y+34),0.62,col,2)
        txt(im,note,(1196,y+52),0.42,DIM,1)
    txt(im,"White line = base rate. A rule that does not clear it carries no information, however sensible it sounds.",(48,730),0.5,DIM,1)
    cv2.imwrite(str(ASSETS/"poolvision-rules.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,92])
    print("wrote poolvision-rules.jpg")

    # Where attribution stands.
    im=np.full((620,1400,3),BG,np.uint8)
    txt(im,"Who took the shot: what each approach could answer",(48,62),0.85,INK,2)
    txt(im,"Out of 34 shots on one recording. Answering is not the same as being right.",(48,100),0.52,DIM,1)
    rows=[("Looking for hands near the ball",5,DIM),
          ("Following the ball's flight back",22,WARN),
          ("Both together",24,GOOD)]
    for i,(lab,v,col) in enumerate(rows):
        y=170+i*110
        cv2.rectangle(im,(48,y),(1180,y+58),(30,34,42),-1)
        cv2.rectangle(im,(48,y),(48+int(1132*v/34),y+58),col,-1)
        txt(im,lab,(62,y+38),0.58,(16,18,24) if v>10 else INK,2)
        txt(im,f"{v} of 34",(1200,y+38),0.6,col,2)
    txt(im,"The rotation test still scores 41% against a 33% coin flip, so correctness is unproven.",(48,560),0.52,DIM,1)
    cv2.imwrite(str(ASSETS/"poolvision-attribution.jpg"),im,[cv2.IMWRITE_JPEG_QUALITY,92])
    print("wrote poolvision-attribution.jpg")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
