?($	?&?D?c?????)??2w-!???!%??C???$	??̈́ @¨tnJ@?}A_ @!W<?J(4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`vOj??`vOj??A?s????Y'1?Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%??C???6<?R???Aj?t???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gx$(?????ׁs??A?1w-!??Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&?W??5?8EGr??A-!?lV??Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????S??@?߾???A?u?????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&S???=,Ԛ???Ad?]K???Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?t?V???N@a???Ag??j+???Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4??????????A_?Q???Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??M?O???A? ?rh???Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?lV}????X9??v???A??K7?A??Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
E???JY?????~?:??A????߾??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(\?????46<?R??A??d?`T??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????9#???Ǻ????A2U0*???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&R???Q????St$???A?K7?A`??Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??u??????(????A??	h"l??Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????8??d?`??A???????Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????~j?t??Aё\?C???Y???<,Ԋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V}??b??X?5?;N??A3ı.n???Y_?Qڋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?p=
ף???L?J???A
ףp=
??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?b?=y???ZB>????A?X?? ??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2w-!????-?????A?>W[????Y???{????*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatC?i?q???!mugp?9?@)?[ A???1?ժ?Α<@:Preprocessing2F
Iterator::Model?J?4??!L??yB@)??&???1????p/8@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ͪ????!?'?G??O@)V}??b??1a#??*@:Preprocessing2U
Iterator::Model::ParallelMapV2?(\?????!hc!-?)@)?(\?????1hc!-?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?$??C??!Q>=?+@)??q????1$?yC?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice$???~???!?????@)$???~???1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ?|a??!H??2@)??\m????1?'?k@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Dio??ɔ?!????@@)Dio??ɔ?1????@@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9I?Yׄ,@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?,?߯?????%?н??-?????!?N@a???	!       "	!       *	!       2$	ڒ+?[???1;?j\????	h"l??!?u?????:	!       B	!       J$	?ݸ?1?????Hm?????<,Ԋ?!'1?Z??R	!       Z$	?ݸ?1?????Hm?????<,Ԋ?!'1?Z??JCPU_ONLYYI?Yׄ,@b Y      Y@q??N*?9G@"?
both?Your program is POTENTIALLY input-bound because 55.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?46.4525% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 