?($	??x??C{?|???lV}???!?"??~j??$	5??Z5@wGQjw@⺘??A@!? !?A?+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??%䃞??6?>W[???Alxz?,C??Y??ܵ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE??????ZӼ??A?46<??Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?"??~j????q????A$(~????Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%??io???T??A???T????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v??????)?Ǻ???Axz?,C??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& c?ZB>??	?^)???A/?$???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ??????{??Pk??A??:M???Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX????.n????A?[ A?c??YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V?????????Alxz?,C??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	R'???????Fx$??A?:pΈ??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?Q??????8EGr???As??A???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%??C???z?):????Avq?-??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&lxz?,C?????߾??AHP?s??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S???٬?\m??AC??6??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&b??4?8????C?l??A???1????Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??H?}???+e?X??A??ʡE??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????mV}??b??A?i?q????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????x????<,Ԛ???AR'??????YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE??m???????A.?!??u??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??{??P???D???J??AtF??_??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}???t$???~??A?h o???Y46<?R??*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???QI??!F?pR??A@)???߾??1???؊?@@:Preprocessing2F
Iterator::Model?Y??ڊ??!??A?%B@)6?;Nё??1W'uG5@:Preprocessing2U
Iterator::Model::ParallelMapV2d?]K???!??<?".@)d?]K???1??<?".@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'?W???!N??<?O@)Qk?w????1     P$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???QI??!?????K@)???QI??1?????K@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?A`??"??!ylE?p,@)?HP???1h8????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??0??!???؊?1@)B>?٬???1c+????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*a2U0*???!?Iݗ?V@)a2U0*???1?Iݗ?V@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?I?5Z@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Z`$˝.??a*N??Ϻ?t$???~??!??q????	!       "	!       *	!       2$	?????4??D???Q.???h o???!$(~????:	!       B	!       J$	????ژ??Ò_?????W[?????!??ܵ???R	!       Z$	????ژ??Ò_?????W[?????!??ܵ???JCPU_ONLYY?I?5Z@b Y      Y@qu2?+BD@"?
both?Your program is POTENTIALLY input-bound because 44.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?40.2051% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 