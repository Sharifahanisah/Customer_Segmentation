	@a��+�?@a��+�?!@a��+�?	���A.f@���A.f@!���A.f@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$@a��+�?;M�O�?A�sF���?Yr�鷯�?*	     �^@2F
Iterator::Model1�Zd�?!�K�`�E@)�lV}��?1�`m�'B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���Q��?!�����8@)p_�Q�?1ڼOq�5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat}гY���?!mާ�d5@)'�����?1	���}�1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��K7��?!v�y��L@)n���?1ڼOq� @:Preprocessing2U
Iterator::Model::ParallelMapV2HP�sׂ?!"XG��)@)HP�sׂ?1"XG��)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU���N@s?!��d��@)U���N@s?1��d��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!�6�S\2@)"��u��q?1�6�S\2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���{�?!amާ�:@)��_�Le?1����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t14.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���A.f@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	;M�O�?;M�O�?!;M�O�?      ��!       "      ��!       *      ��!       2	�sF���?�sF���?!�sF���?:      ��!       B      ��!       J	r�鷯�?r�鷯�?!r�鷯�?R      ��!       Z	r�鷯�?r�鷯�?!r�鷯�?JCPU_ONLYY���A.f@b 