	?@I??6@?@I??6@!?@I??6@	r{=Ô???r{=Ô???!r{=Ô???"w
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
)????B      ??!       J	???s???????s????!???s????R      ??!       Z	???s???????s????!???s????b      ??!       JGPUYr{=Ô???b q?nc??"@y??̧XW@