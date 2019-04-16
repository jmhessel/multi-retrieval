Code for Unsupervised Discovery of Multimodal Links in Multi-Image/Multi-Sentence Documents

## Requirements
This code requires python3 and several python
libraries. You can install the python requirements with:

```
pip3 install -r requirements.txt
```

Also --- it helps performance to initialize the word embedding
matrices with word2vec embeddings. You can download those embeddings
[here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
(be sure to extract them). When you run the training command, it is
recommended to use the option `--word2vec_binary XXX` where XXX is the
path to the extracted/downloaded word embeddings.

## How to run

### Preparing the dataset

The training script takes three inputs:

1. A json of training/validation/test documents. This json stores a dictionary with three keys: `train`, `val`, and `test`. Each of the keys maps to a list of documents. A document is a list containing 3 things: `[list_of_images, list_of_sentences, metadata]`.
  - `list_of_images` is a list of `(identifier, label_text_idx)` tuples, where the identifier is the name of the image, and `label_text_idx` is an integer indicating the index of the corresponding ground-truth sentence in `list_of_sentences`. If there are no labels in the corpus, this index can be set to `None`. If there are labels, but this particular image doesn't correspond to a sentence, you can set the index to `-1`.
  - `list_of_sentences` is a list of `(sentence, label_image_idx)` tuples, where sentence is the sentence, and `label_image_idx` is an integer indicating the index of the corresponding ground-truth image in `list_of_images`. If there are no labels in the corpus, this index can be set to `None`. If there are labels, but this particular image doesn't correspond to a sentence, you can set the index to `-1`.
  - `metadata` is an optional document identifier.
2. A json mapping image ids (see `list_of_images`) to row indices in the features matrix.
3. An image feature matrix, where `matrix[id2row[img_id]]` is the image feature vector corresponding to the image with image id `img_id` and `id2row` is the dictionary stored in the previously described json mapping file.

Here is an example document from the MSCOCO dataset.
```
[[['000000074794', -1],
  ['000000339384', 9],
  ['000000100064', -1],
  ['000000072850', 8],
  ['000000046251', -1],
  ['000000531828', -1],
  ['000000574207', 0],
  ['000000185258', 5],
  ['000000416357', 1],
  ['000000490222', -1]],
 [['Two street signs at an intersection on a cloudy day.', -1],
  ['A man holding a tennis racquet on a tennis court.', -1],
  ['A seagull opens its mouth while standing on a beach.', -1],
  ['a man reaching up to hit a tennis ball', -1],
  ['A horse sticks his head out of an open stable door. ', -1],
  ['Couple standing on a pier with a lot of flags.', -1],
  ['A man is riding a skateboard on a ramp.', -1],
  ['A man on snow skis leans on his ski poles as he stands in the snow and '
   'gazes into the distance.',
   -1],
  ['a close up of a baseball player with a ball and glove', -1],
  ['four people jumping in the air and reaching for a frisbee.', -1]],
 'na']
```

The [image with ID](http://cocodataset.org/#explore?id=339384)
`000000339384` in the MSCOCO dataset corresponds to the caption with
sentence with index 9 in this document, "four people jumping in the
air and reaching for a frisbee.". The underlying graph is undirected,
so the labels are stored only in the image list (though, if you like,
you could redundantly store them on the text-side). For the MSCOCO
dataset, the metadata is un-used.

The exact train/val/test splits we used, along with pre-extracted
image features, are available for download (see below). You can download
these and extract them in the `data` folder.

### Extracting image features for a new dataset

If you would like to extract image features for a new dataset, there
are a number of existing codebases for that, depending on what neural
network you would like to use. We have included the script that we
used to do that, if you'd like to use ours. In particular, you should:

1. Get all of the images of interest into a single folder. Your images should all have unique filenames, as the scripts assume that, e.g., the name of the `jpg` file is the identifier, e.g., `my_images/000000072850.jpg`'s identifier will be `000000072850`.
2. Create a text file with the full paths of each image
3. Call `python3 image_feature_extract/extract.py [filenames text file] extracted_features`
4. Call `python3 make_python_image_info.py extracted_features [filenames text file]`

This will output a feature matrix (in npy format) and an id2row json
file. These are two of the three arguments. Note --- you may need to
modify `make_python_image_info.py` if your images have different
folders, or if you have multiple images with the same name but
different extensions, e.g., `id.jpg` and `id.png` will both erronously
be mapped to `id`. I may add support for this later (in addition to
cleaning up these scripts...).

## How to run the code

An example training command for the mscoco dataset with reasonable settings is:
```
python3 train_doc.py data/mscoco/docs.json \
--image_id2row data/mscoco/id2row.json \
--image_features data/mscoco/features.npy \
--word2vec_binary data/GoogleNews-vectors-negative300.bin \
--cached_word_embeddings mscoco_cached_word_embs.json \
--print_metrics 1 \
--output mscoco_results.pkl
```

note that even though metrics are printing during training if you use
`--print_metrics 1`, there is no early stopping/supervision happening
on the labels during training.

you can run this to get more information about particular training options
```
python3 train_doc.py --help
```

From the paper, here's an example of running with hard negative
mining, the AP similarity function, and 20 negative samples
```
python3 train_doc.py data/mscoco/docs.json --image_id2row data/mscoco/id2row.json
--image_features data/mscoco/features.npy
--word2vec_binary data/GoogleNews-vectors-negative300.bin \
--cached_word_embeddings mscoco_cached_word_embs.json \
--print_metrics 1 \
--output mscoco_results.pkl \
--sim_mode AP \
--docs_per_batch 21
```

## How to reproduce the results of the paper:

The datasets we use with specific splits/pre-extracted image features
are available for download. If you are just using the datasets, please
cite the original creators of the datasets. Furthermore, all datasets
are subsets of their original creator's releases; please use the
versions from the original links if you are looking for more complete
datasets!

- MSCOCO ([original source](http://cocodataset.org/#home)) [link](https://drive.google.com/open?id=1LGqUst-BB8N4nFPNGHD0uVa3x_cAZ7UV)
- DII ([original source](http://visionandlanguage.net/VIST/dataset.html)) [link](https://drive.google.com/open?id=1zFouzVhXvnK19zv3AYT-wZJt8SFTcRXY)
- SIS ([original source](http://visionandlanguage.net/VIST/dataset.html)) [link](https://drive.google.com/open?id=1MN6gPGhymAHvPJL6dRTu-VbXfYlI0L7-)
- DII-Stress ([original source](http://visionandlanguage.net/VIST/dataset.html)) [link](https://drive.google.com/open?id=1vLOMftRh8U5r3sn29X2l8XxVXQppsLYS)
- RQA ([original source](https://hucvl.github.io/recipeqa/)) [link](https://drive.google.com/open?id=1BbD1OnV4h02QUk1eZT1hFWWKlDwUyz3O)
- DIY [link](https://drive.google.com/open?id=1EdgL2VYrVTLccP8wHpynpFhv3PNuZiOv)
- WIKI ([original source](https://www.imageclef.org/wikidata)) [link](https://drive.google.com/open?id=1Ecb1LkTXX4sskx-PLB2o3vMru-8I8rEy)

In addition, we have included scripts that generate the exact training commands executed in the paper itself. These are located in the paper_commands directory.
