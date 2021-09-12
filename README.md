# StreamHover: Livestream Transcript Summarization and Annotation

We provide the source code for the paper **"[StreamHover: Livestream Transcript Summarization and Annotation]()"**, accepted at EMNLP 2021. If you find the code useful, please cite the following paper.

```
@inproceedings{cho-et-al:2021,
 Author = {Sangwoo Cho, Franck Dernoncourt, Tim Ganter, Trung Bui, Nedim Lipka, Walter Chang, Hailin Jin, Jonathan Brandt, Hassan Foroosh and Fei Liu},
 Title = {StreamHover: Livestream Transcript Summarization and Annotation},
 Booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
 Year = {2021}}
```

## Dependencies

We suggest the following environment:

  - [Anaconda](https://anaconda.org/)
  - [Python (v3.6)](https://www.anaconda.com/download/)
  - [Pytorch (v1.4)](https://pytorch.org/get-started/locally/)
  - [Pyrouge](https://pypi.org/project/pyrouge/)
  - [Huggingface Transformer](https://github.com/huggingface/transformers)
  - Create the same environment with `conda env create --file environment.yml`


## Behance Dataset

  - Download data from [HERE](https://drive.google.com/file/d/1kMmMX7ceYLOZuhdsgi_Qahc269Bpipha/view?usp=sharing) 
        - Unzip the file and move to `path/to/your/data/folder`           
        - Each pickle file (`Behance_train.pkl`, `Behance_val.pkl`, `Behance_test.pkl`) contains a list of the following data, which is based on a 5-min. transcript      
            ```
            (
            	List[dict],    # transcript of 5 min. clip
            	str,           # abstractive summary
            	List[int],     # extractive summary (indices of each utterance, 0-based)
            	int,           # unique video ID
            	int,		   # unique clip ID (e.g. 0 means 0-5 min. clip, 1 means 5-10 min. clip)
            	str,		   # video title
            	str,		   # video url
            	str			   # transcript url
            )
            ```
      
        - Transcript dictionary above contains the following data
            ``` 
            {
            	'display': str,      # utterance
            	'offset': float,     # start time of the utterance
            	'duration': float    # duration of the utterance
            }
            ```
      
        - In the paper, we used the following 3,884 clips in train / 728 clips in val / 809 clips in test for experiments. However, due to the privacy issue of two videos in the training set, we remove them and provide the following data.
            - train: 3,860 clips from 318 videos (24 clips are removed from 2 videos)
            - val: 728 clips from 25 videos
            - test: 809 clips from 25 videos

## Train / Test Models

- Trained model can be downloaded from [HERE](https://drive.google.com/file/d/1rn-OsnnvNUkM-iSF9aEB6rEEqn9A_w2Z/view?usp=sharing) 
  - Download and move it to the folder `/models/c1024_e100`
    - codebook size: 1024, convolution filter size: 100
  - Please refer to `src/commands.sh` for command examples.

## Generate a Clip-Level Summary

- Please refer to `src/commands.sh` for a command example.
- A summary output file (`*.json`) will be generated in `results` folder.

## Generate a Video-Level Summary

- Please refer to `src/commands.sh` for a command example.
- Summary utterances are selected from each valid 5 min. clip in a video, and each selected utterances are merged for a video-level summary.
- You can use the following arguments to generate one.
  - `video_inference_id`: video ID for inference (refer to [`videoID_split.csv`](https://drive.google.com/file/d/1M8jdj35zuJ0V_SoOgPQ9KNWnKcVTo9PF/view?usp=sharing) to obtain an index number for a video that you want to generate a video-level summary e.g. for row 9 in the file, index=7, video id=16, split=train, you need to set this argument with value of `7` for one of videos in the training set)
  - `video_inf_min_sent`: summary generation is skipped if the number of utterances in any 5 min. clip is less than this value
  - `num_sum_sent`: number of summary utterances for each 5 min. clip
- A summary output file (`*.json`) and transcript file (`*.json`) will be generated in `results` folder.
