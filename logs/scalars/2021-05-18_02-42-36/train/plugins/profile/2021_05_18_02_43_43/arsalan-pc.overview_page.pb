?	?w?~?F6@?w?~?F6@!?w?~?F6@	LL?2@???LL?2@???!LL?2@???"w
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
??X??:	Pr?Mdf??Pr?Mdf??!Pr?Mdf??B      ??!       J	???4??????4???!???4???R      ??!       Z	???4??????4???!???4???b      ??!       JGPUYLL?2@???b q?+?6??@ynK???PW@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?mi!s???!?mi!s???0"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???N??!$-P?K??0"F
(gradient_tape/sequential/conv2d/ReluGradReluGrad}/&????!?????p??"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad??+?4???!??G^??":
sequential/conv2d_1/Relu_FusedConv2D?zO????!?p??"??"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputN???#???!??-R{x??0"-
IteratorGetNext/_1_Send??z?????!????Yu??"?
hgradient_tape/sequential/spatial_dropout2d/dropout/Mul-0-0-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown?-?Z?^??!?	Z?8???"A
(sequential/spatial_dropout2d/dropout/MulMul5???TY??!?U0c???"O
6gradient_tape/sequential/spatial_dropout2d/dropout/MulMulc$P?%??!?D,=???Q      Y@Yc'vb'v2@a'vb'vbT@qPR!q%b??yH?o?"???"?

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