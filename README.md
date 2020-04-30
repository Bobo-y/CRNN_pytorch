最近学习OCR方面的检测, 以前只使用别人写好的CRNN来微调过, 今天参照之前的Keras-CRNN和CRNN论文, 
使用VGG16作为backbone实现了CRNN. 没有GPU机器, 自己造了100多张图片在CPU上来测试模型能否拟合, 能够训练出来.

