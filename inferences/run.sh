DETDIR=./X_101_32x8d
python image_bbox_vg_extraction.py $DETDIR/config.yml $DETDIR/model_final.pth \
  data/images data/img_names.npy data/bbox_fts

python image_feature_extraction_given_bbox.py $DETDIR/config.yml $DETDIR/model_final.pth \
  data/images/ data/img_names.npy data/fts_given_bbox/img_bboxes.json data/fts_given_bbox/

