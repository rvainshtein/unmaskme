# unmaskme
## TODO
Data (masktheface):
- [X] run and generate images 
- [X] download the full data from drive (128x128)
- [ ] run on the full data and generate the train set
- [ ] integrate on the fly into training?

Generation (cyclegan or something else?):
- [ ] run inference on example data
- [ ] train on a dataset that was already trained on
- [ ] train on our data

## pix2pix

### running train usage
python train.py --dataroot ./datasets/maskface/data_combined --model pix2pix --name face2mask --print_freq 50 --num_threads 12

### running test usage
python test.py --dataroot ./../our_data/data_combined --model pix2pix --name face2mask --direction BtoA --results_dir ./results/