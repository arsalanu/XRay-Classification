	?w?~?F6@?w?~?F6@!?w?~?F6@	LL?2@???LL?2@???!LL?2@???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?w?~?F6@?o??}??1??ǵ??4@A??
??X??IPr?Mdf??Y???4???*	D?l??QT@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-???!HRsy??A@)???N]??1d??zy>@:Preprocessing2F
Iterator::Model y?P????!??i#D@)???B????1IC?
k>6@:Preprocessing2U
Iterator::Model::ParallelMapV29_??????!?G<??1@)9_??????1?G<??1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?? v???!h??c2$@)?? v???1h??c2$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh??n???!?5T}e1@)??.??x?16ƫu??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??x#???!L????M@)5?l?/r?1D?ȡ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3???ySq?!?<?{?@)3???ySq?1?<?{?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq??]P??!6&T??2@)HP?s?R?1OoL4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9LL?2@???I?+?6??@QnK???PW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?o??}???o??}??!?o??}??      ??!       "	??ǵ??4@??ǵ??4@!??ǵ??4@*      ??!       2	??
??X????
??X??!??
??X??:	Pr?Mdf??Pr?Mdf??!Pr?Mdf??B      ??!       J	???4??????4???!???4???R      ??!       Z	???4??????4???!???4???b      ??!       JGPUYLL?2@???b q?+?6??@ynK???PW@