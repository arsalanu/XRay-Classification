?	^??I??Y@^??I??Y@!^??I??Y@	z/?{??@z/?{??@!z/?{??@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6^??I??Y@ߤiP4 @1J?i?)M@Aіs)????Im????@@Y?*?)@*	X9?hǻ@2F
Iterator::Model??z0?@!`??I?J@)?m?	@1?=d??F@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??`?H???!Z?U8=(@)??`?H???1Z?U8=(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??,z????!BA???#@)??,z????1BA???#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????Z
@!????)G@)???????13W|m;
"@:Preprocessing2U
Iterator::Model::ParallelMapV2f?O7P`??!?]*z& @)f?O7P`??1?]*z& @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?W˝????!*B????5@)^gE?D??1??N?7?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ަ?????!????R/@)b?A
?B??1?ð?M@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??q?dO??!??6;m.@)?g	2*??1?8????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?31.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9z/?{??@I5???5D@Q?i??BL@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ߤiP4 @ߤiP4 @!ߤiP4 @      ??!       "	J?i?)M@J?i?)M@!J?i?)M@*      ??!       2	іs)????іs)????!іs)????:	m????@@m????@@!m????@@B      ??!       J	?*?)@?*?)@!?*?)@R      ??!       Z	?*?)@?*?)@!?*?)@b      ??!       JGPUYz/?{??@b q5???5D@y?i??BL@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???"??!???"??0":
sequential/conv2d_1/Relu_FusedConv2DT]????!Bx?????"F
(gradient_tape/sequential/conv2d/ReluGradReluGrad?R?n???!???|????"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput^??T>??!??hl??0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh?2?c???!?W?~@??0"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad(? ???!Xe?D???"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?%l??!???g??0"-
IteratorGetNext/_1_Send?????ǡ?!љ??ʠ??":
sequential/conv2d_2/Relu_FusedConv2D?g??k\??!??\X???"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Z?d???!Cp?????0Q      Y@Y/?袋.2@au?E]tT@qs?)???y??5??'w?"?

both?Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?31.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 