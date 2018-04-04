#!/bin/sh

rm -rf deploy
mkdir ./deploy

cp ./* ./deploy
cp -r alphabets encoders helpers neural_network scripts ./deploy

echo "Submitting training job"
cf sapml job submit -f ./train.sapml ./deploy

echo "Training job submitted"
