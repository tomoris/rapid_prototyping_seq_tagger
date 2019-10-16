# Rapid Prototyping of Domain-Specific Named Entity Tagger
Implementation of PYHSCRF (tomori et al., 2019) with python and golang. PYHSCRF only requires a few seed terms, in addition to unannotated corpora for named entity recognition. It permits the iterative and incremental design of named entity classes for new domains. PYHSCRF is a hybrid of a generative model named PYHSMM and a semi-Markov CRF-based discriminative model, which play complementary roles in generalizing seed terms and in distinguishing between NE chunks and non-NE words.

## Prerequisites
```
Python 3
Pytorch 1.0.0

Go 12.5
github.com/tomoris/PYHSMM
```

## Installing
```
python setup.py build (TODO)
```

## Usage
In training:  
` python rapid_prototyping_seq_tagger/main.py --mode train --config sample/config.py`  
In predicting:  
` python rapid_prototyping_seq_tagger/main.py --mode predict --load_model model_dir/tagger.model`  

## Decoding Model and Data Format
This supports both Markov CRF and semi-Markov CRF. These can be trained on partially annotated data  
    - BIESO tagging scheme should be used if you train a model with semi-Markov CRF

## Evaluation


## References
- (tomori et al., 2015) https://easychair.org/publications/preprint/LlQj

## License