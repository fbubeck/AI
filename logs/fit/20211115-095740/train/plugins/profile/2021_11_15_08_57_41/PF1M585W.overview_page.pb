?($	??V/AJ???%????????MbX??!T㥛? ??$	????M@E.lUW?
@"?`Iۂ@!?q?q0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?.n?????:pΈҾ?A@a??+??Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A???}8gD??A=,Ԛ???Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'?????9??m4???A??y?):??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ????(\????A?A`??"??Y?z?G???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S????8gDio??A鷯???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S??l	??g???A?I+???YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ffffff?????K7???A?\?C????YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?C?????X?2ı.??AQ?|a2??Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ??bX9????AY?8??m??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???1???????1????A??C?l???Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
???????ffffff??A*:??H??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2w-!????v??/??Ae?X???Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?	???[ A?c??A>?٬?\??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ڊ?e???pΈ?????AF%u???Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??v????$(~????A??????YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??????Mb??AK?=?U??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s???????? ?r??AEGr????YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY????MbX9??A?H?}8??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4??Gx$(??A*??D???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??M?O???A!?rh????Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??MbX????|?5^??A-!?lV??Y?|a2U??*	43333W?@2F
Iterator::Model?A`??"??!?%??D@)?m4??@??14x???u;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatı.n???!??P?^K=@)?o_???1?vTA:@:Preprocessing2U
Iterator::Model::ParallelMapV2?*??	??!ܥ{?$)@)?*??	??1ܥ{?$)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?7??d???!p?g??M@)??ʡE???1D?a?Y?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?q??????!?:?? @)?q??????1?:?? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??????!^$'?c?0@)K?=?U??1??5?Bc @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapd]?Fx??!&??^:]4@)?v??/??1D?6e??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?z6?>??!Pf2WP@)?z6?>??1Pf2WP@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?3??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Й?&??????D#O?????|?5^??!9??m4???	!       "	!       *	!       2$	?(݌B???O?????-!?lV??!?I+???:	!       B	!       J$	8??դ??? fv?^????Pk?w??!??q????R	!       Z$	8??դ??? fv?^????Pk?w??!??q????JCPU_ONLYY?3??@b Y      Y@q?,??q?B@"?
both?Your program is POTENTIALLY input-bound because 48.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.7925% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 