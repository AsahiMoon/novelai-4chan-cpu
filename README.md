# introduction

change  novelai-4chan ver to run on cpu and ram

# docker

## build

`docker build -t novelai_image:v1 .`

## run

`docker run  -id -p 6969:6969 -v your_absolute_path/models:/novelai/models novelai_image:v1 ./run.sh`

# tips

- **models** file need to download from other way
- use the**animefull-final-pruned** model may need more than 6GB ram
- other**colab** ver maybe more useful
