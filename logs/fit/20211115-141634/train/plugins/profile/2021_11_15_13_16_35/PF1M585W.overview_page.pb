?($	??#Fdo??jy??m????/?$??!ŏ1w-!??$	D?xcP;@?Z??@O??b?@!?IƢf+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$W[???????0?*??A?c]?F??Y[B>?٬??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j????٬?\m??Ah"lxz???YV????_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-!???6?[ ??A??&???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?h o????U??????A3ı.n???Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)\???(???+e?X??A? ?	???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F????x????(????A?D???J??Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??JY?8???):????A8??d?`??Y(~??k	??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~??k	????D?????AY?? ???Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M???-???????A???&??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	+??ݓ????J?4??A?ZB>????Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??? ?r??:#J{?/??AY?? ???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'1?Z??w-!?l??A???镲??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-!?lV????y?)??A?I+???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&???W?/?'??A?:pΈ??YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2w-!???a??+e??A^K?=???Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`?????QI??&??A6<?R???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L?
F%u??h"lxz???A{?/L?
??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=,Ԛ???jM????A?q?????Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\??d]?Fx??A$???????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F??]?C?????A?A?f???Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??/?$????N@a??A??#?????Y0*??D??*	33333{?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!k|0&?@@)I??&??1??@?/I>@:Preprocessing2F
Iterator::Model      ??!F????B@)?0?*??1/-??"?6@:Preprocessing2U
Iterator::Model::ParallelMapV2?A?f????!??ðR-@)?A?f????1??ðR-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[????<??!???SBcO@)0*??D??1???#?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatea2U0*???!YbU0?~.@)]?Fx??1DG?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicef?c]?F??!m?cn?@)f?c]?F??1m?cn?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H.???!Es4?*)4@)??MbX??1c' ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*%u???!K_?X@)%u???1K_?X@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?b??b?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	ù\???????E??????N@a??!?6?[ ??	!       "	!       *	!       2$	??????Us?š????#?????!h"lxz???:	!       B	!       J$	ڻmNO??????6!??+??????!	?c???R	!       Z$	ڻmNO??????6!??+??????!	?c???JCPU_ONLYY?b??b?@b Y      Y@qw?wB@"?
both?Your program is POTENTIALLY input-bound because 56.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?36.9304% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 