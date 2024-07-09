python demo/demo_with_text.py --chunk_size 4 \
--img_path ../data/JPEGImages/lucia \
--sam_variant sam_hq \
--size 512 \
--output ../data/Annotations/lucia_out --prompt person

python parse_mask.py ../data/Annotations/lucia_out/ lucia

python demo/demo_with_text.py --chunk_size 4 \
--img_path ../data/JPEGImages/stroller \
--sam_variant sam_hq \
--size 512 \
--output ../data/Annotations/stroller_out --prompt woman.stroller

python parse_mask.py ../data/Annotations/stroller_out/ stroller