	EGr���?EGr���?!EGr���?	�z���@�z���@!�z���@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$EGr���?ڬ�\m��?AH�z�G�?YbX9�ȶ?*	    �d@2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�#�����?!�.�袋@@)�#�����?1�.�袋@@:Preprocessing2F
Iterator::Model�z6�>�?!�>���;@)vOjM�?1�d�M6�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL7�A`�?!      4@)y�&1��?1���>�0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr�鷯�?!N6�d�MG@)��ͪ�Ֆ?1|��+@:Preprocessing2U
Iterator::Model::ParallelMapV2� �	�?!������@)� �	�?1������@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu���?!|��R@)_�Q�{?1|��|@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{�G�zt?!>���>@){�G�zt?1>���>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ�|a�?!�|��5@)�����g?1|���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t14.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�z���@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ڬ�\m��?ڬ�\m��?!ڬ�\m��?      ��!       "      ��!       *      ��!       2	H�z�G�?H�z�G�?!H�z�G�?:      ��!       B      ��!       J	bX9�ȶ?bX9�ȶ?!bX9�ȶ?R      ��!       Z	bX9�ȶ?bX9�ȶ?!bX9�ȶ?JCPU_ONLYY�z���@b 