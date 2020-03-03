## Usage
### data
data can be downloaded from [here](https://drive.google.com/open?id=1aR3WlSRdF-Av8Eup0J8OhNCkHoiMogOH)

### train+val
```python
python3 training_ptr_gen/train_with_eval.py >& ./path/to/logfile &
```
### test
```python
python3 training_ptr_gen/decode.py ./log/train_[trainID]/model/model_[iterationNum]_[weightID] >& ./log/decode_log.model_[iterationNum]_[weightID] &
```

I recommend using tmux or byobu for management.
