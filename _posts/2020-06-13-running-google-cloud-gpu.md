# Running on Google Cloud GPU From Colab

[https://cloud.google.com/sdk/install](https://cloud.google.com/sdk/install)

### Login into google cloud from your terminal

gcloud init

### Set your project

gcloud config set project java-ref 
gcloud config set account java-ref

*replace java-ref with your project name*  

### Start instance using GPU


gcloud beta compute instances create torchvm3 \
  --zone=us-central1-f \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --machine-type="n1-standard-1" \
  --scopes storage-rw


gcloud beta compute instances create torchvm3 \
  --zone=us-central1-a \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=2" \
  --metadata="install-nvidia-driver=True" \
  --machine-type="n1-standard-4" \
  --scopes storage-rw \
  --preemptible

  remove preeptible if you don't get instance

  also change zone depending on availability of GPU


### SSH Into The instance 

gcloud compute ssh --project java-ref --zone us-west1-b torch2-vm -- -L 8080:localhost:8080

ssh -nNT -L 8888:localhost:8888 node@176.9.137.77 -p 7002




  ### Setup your instance for connection with colab


jupyter serverextension enable --py jupyter_http_over_ws

jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0 --JupyterWebsocketPersonality.list_kernels=True --NotebookApp.disable_check_xsrf=True --no-browser --NotebookApp.password='' --no-mathjax &



gcloud compute project-info add-metadata \
    --metadata enable-oslogin=TRUE
# all os login to all instances





gcloud compute instances attach-disk torchvm3 --disk torch --zone us-central1-f

gcloud beta compute ssh --project java-ref --zone us-central1-a torchvm3 -- -L 8080:localhost:8080

sudo lsblk
sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb



sudo mkdir -p /home/jupyter/drive
sudo mount -o discard,defaults /dev/sdb /home/jupyter/drive
sudo chmod a+w /home/jupyter/drive

gcloud compute instances stop torchvm3 --zone=us-central1-a
gcloud compute instances delete torchvm3 --zone=us-central1-a


gcloud auth login
gcloud config set project recruitai-266705
gsutil ls
gsutil -m cp -r recruit-ner-bert-flair-augment/ gs://recruitaiwork/recruit-ner-flair-augment