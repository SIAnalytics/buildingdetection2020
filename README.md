# Building Detection2020
2020년 국가 위성영상 AI 데이터 구축 사업의 일환으로 만들어진 코드입니다.

# Model
![전체 모델 구조](./images/model.png)

# 데이터셋 소개
- 인공위성: 다목적실용위성 (KOMPSAT) 3호/3A호  (아리랑위성)
- Pixel size: (1024, 1024)
- Classes: 소형시설(1), 아파트(2), 공장(3), 중형단독시설(4), 대형시설(5)
- Label type: geojson (영상의 위/경도, 건물 polygon좌표, classes 등)


# Weight file download
- DeeplabV3+ResNet101
- DeeplabV3+ResNet50
- FCN+ResNet101
- LINK: https://www.dropbox.com/sh/ja28r1tir8varvi/AADaGnamlFcGO-fZ7mumMQ-aa?dl=0
