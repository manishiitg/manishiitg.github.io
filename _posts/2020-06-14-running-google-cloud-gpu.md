# Running on Google Cloud GPU From Colab

[https://cloud.google.com/sdk/install](https://cloud.google.com/sdk/install)

### Login into google cloud from your terminal

```sh
gcloud init
```

### Set your project

```
gcloud config set project java-ref 
gcloud config set account java-ref
```

*replace java-ref with your project name*  

### Start instance using GPU

```
gcloud beta compute instances create torchvm3 \
  --zone=us-central1-f \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --machine-type="n1-standard-1" \
  --scopes storage-rw
```

```
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
```


  remove preeptible if you don't get instance
  also change no of gpus as you need

  also change zone depending on availability of GPU

### Attach a disk to save progress


```
gcloud compute disks create torch --zone us-central1-a --type pd-ssd

gcloud compute instances attach-disk torchvm3 --disk torch --zone us-central1-a
```

attach a disk so you can save your checkpoints and resume. 
if you are using preemtable instance this is best to do as instance can go away anytime.


### SSH Into The instance 

```
gcloud compute ssh --project java-ref --zone us-central1-a torchvm3 -- -L 8080:localhost:8080
```
Run below commands on the gpu instance

nvidia-smi

*check your gpu*

jupyter serverextension enable --py jupyter_http_over_ws

jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0 --JupyterWebsocketPersonality.list_kernels=True --NotebookApp.disable_check_xsrf=True --no-browser --NotebookApp.password='' --no-mathjax &



Connect your google colab to localhost:8080

[https://research.google.com/colaboratory/local-runtimes.html](https://research.google.com/colaboratory/local-runtimes.html)


### Mount your disk

sudo lsblk
sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb

sudo mkdir -p /home/jupyter/drive
sudo mount -o discard,defaults /dev/sdb /home/jupyter/drive
sudo chmod a+w /home/jupyter/drive


another tip is to run "top" on ssh so that connection is active. else sometimes i have noticed connection just stops
also if you don't have a stable internet

### Delete Instance (Important Step)

gcloud compute instances stop torchvm3 --zone=us-central1-a
gcloud compute instances delete torchvm3 --zone=us-central1-a


### Once Training Upload to Cloud Storage

optional step

gcloud auth login
gcloud config set project recruitai-266705
gsutil ls
gsutil -m cp -r recruit-ner-bert-flair-augment/ gs://recruitaiwork/recruit-ner-flair-augment