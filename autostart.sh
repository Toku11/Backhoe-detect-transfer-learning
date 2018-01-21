mkdir data1
mkdir data
mkdir images
mkdir annotations
mkdir eval
echo "Creando Dataset..."
python dataset_merge.py
python xml_to_csv.py
mv Skycatch.csv ./data1/
echo "Creando train y val"
python splt.py
mv train_labels.csv ./data1/
mv test_labels.csv ./data1/
echo "Creando tf_record"
python generate_tfrecord.py --csv_input=data1/train_labels.csv  --output_path=train.record
python generate_tfrecord.py --csv_input=data1/test_labels.csv  --output_path=test.record
cp test.record ./data1/
mv test.record ./data/
cp train.record ./data1/
mv train.record ./data/
