	T㥛� �?T㥛� �?!T㥛� �?	@k���@@k���@!@k���@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$T㥛� �?��	h"l�?A"��u���?Y��:M�?*	hfffff\@2F
Iterator::Model��6��?!as �
�G@)bX9�Ȧ?1��@��C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF%u��?!�'�K=7@)46<�R�?16�03@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�b�=y�?!=�]�	5@)���<,�?1�RO�oW1@:Preprocessing2U
Iterator::Model::ParallelMapV2�j+��݃?!z2~��!@)�j+��݃?1z2~��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B�i�?!���.�$J@)lxz�,C|?1w�'�K@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP�s�r?!P�o�z2@)HP�s�r?1P�o�z2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4q?!z2~�ԓ@)�J�4q?1z2~�ԓ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���QI�?!�6-9@)/n��b?1��.�d��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Ak���@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��	h"l�?��	h"l�?!��	h"l�?      ��!       "      ��!       *      ��!       2	"��u���?"��u���?!"��u���?:      ��!       B      ��!       J	��:M�?��:M�?!��:M�?R      ��!       Z	��:M�?��:M�?!��:M�?JCPU_ONLYYAk���@b 