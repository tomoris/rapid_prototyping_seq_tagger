
# TODO


# Requirement
```
Python 3
Pytorch 1.0.0
```

# Usage
In training: ` python rapid_prototyping_seq_tagger/main.py --mode train --config sample/config.py`  
In predicting: ` python rapid_prototyping_seq_tagger/main.py --mode predict --load_model model_dir/tagger.model`  

# Decoding Model and Data Format
This supports both Markov CRF and semi-Markov CRF. These can be trained on partially annotated data  
    - BIESO tagging scheme should be used if you train a model with semi-Markov CRF
