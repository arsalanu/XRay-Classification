?	?@I??6@?@I??6@!?@I??6@	r{=Ô???r{=Ô???!r{=Ô???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?@I??6@?(?[Z???1?熦?X5@A??K???IE?
)????Y???s????*	m????V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-??!??????B@)?v?ӂ??1??g+?@@:Preprocessing2U
Iterator::Model::ParallelMapV2?lscz?!?? P?4@)?lscz?1?? P?4@:Preprocessing2F
Iterator::Model?7?ܘ???!???f?ZB@)h??n???1?"???/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateԸ7?a???!peTz?y3@).??T???1????x$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicezpw?n???!???e?z"@)zpw?n???1???e?z"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorYLl>?u?!????@@)YLl>?u?1????@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip
???리?!l'??O@)	?^)?p?1??E|1?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJӠh???!;???y?4@)]?????Q?1?̱YL???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9r{=Ô???I?nc??"@Q??̧XW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(?[Z????(?[Z???!?(?[Z???      ??!       "	?熦?X5@?熦?X5@!?熦?X5@*      ??!       2	??K?????K???!??K???:	E?
)????E?
)????!E?
)????B      ??!       J	???s???????s????!???s????R      ??!       Z	???s???????s????!???s????b      ??!       JGPUYr{=Ô???b q?nc??"@y??̧XW@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ư2????!?ư2????0"F
(gradient_tape/sequential/conv2d/ReluGradReluGrado"?+???!??LڈE??"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?n
L?@??!??O-????0"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad????A??!????3??":
sequential/conv2d_1/Relu_FusedConv2D{P(?bl??!??H????"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?TvY"???!0?????0"-
IteratorGetNext/_1_Send?U\??!?η??"A
(sequential/spatial_dropout2d/dropout/MulMul??? ??!???=)???"Q
8gradient_tape/sequential/spatial_dropout2d/dropout/Mul_1Mul?DM?????!D;e?????"C
*sequential/spatial_dropout2d/dropout/Mul_1Mulr	????!????t??Q      Y@Yc'vb'v2@a'vb'vbT@qV	?B%?@y?c$q3+??"?

both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 