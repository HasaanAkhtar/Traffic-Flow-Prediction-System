# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19
- xlrd 1.2.0

## Train the model

**Run command below to train the model:**

```
python train_intersection.py --model model_name
```

You can choose "lstm", "gru"  as arguments. The ```.h5``` weight file was saved at model folder.


## Experiment

Data are obtained from VicRoad for the city of Boroondara. Data contained only the traffic flow data (the number of cars passing an intersection every 15 minutes)
	
	device: GTX 1080ti
	dataset: Boroondara Council 15min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 1024 


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


Intersection - MAROONDAH_HWY W of UNION_RD
Lag - 1 
| Metrics | MAE | MSE | RMSE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :----------------------: |
| LSTM | 12.84 | 338.93 | 18.41 | 0.91| 0.92|
| GRU | 12.82 | 374.45 | 19.35 |  0.95 | 0.95|

![evaluate](/images/eva-one-intersection.png)

Intersection - MAROONDAH_HWY W of UNION_RD
Lag - 12 
| Metrics | MAE | MSE | RMSE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :----------------------: |
| LSTM | 11.8 | 273.91 | 16.55 | 0.93| 0.93|
| GRU | 11.5 | 252.93 | 15.90 |  0.94 | 0.94|

![evaluate](/images/eva-one-intersection-lag.png)
## Reference

	@article{SAEs,  
	  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},  
	  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
	  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
	  year={2015}
	}
	
	@article{RNN,  
	  title={Using LSTM and GRU neural network methods for traffic flow prediction},  
	  author={R Fu, Z Zhang, L Li},
	  journal={Chinese Association of Automation, 2017:324-328},
	  year={2017}
	}



