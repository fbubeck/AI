?($	%??h>???x?Q-????????!46<?R??$	R????)@s??/&?@'?;?Q?@!W? ?4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$P?s???+??????A]?C?????Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??T???????#?????Aё\?C???Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Ǻ?????Ax$(~??Y䃞ͪϥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B?f??j??????????A???????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s?????bX9????A6<?R???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A????T???N??AZ??ڊ???Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???????ʡE??AP??n???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ˡE?????d?]K???A_?Q???Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l??Q?|a2??A8??d?`??Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?5^?I???5?;N???A2w-!???YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
$(~????[????<??A???(???Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????ǘ?????Aj?q?????Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B??R'??????A?k	??g??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	???????H??A\ A?c???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?MbX9??lxz?,C??Az?,C???Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q???????!??u???A????????YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF??io???T??A-C??6??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?):????%䃞??A?}8gD??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R???v??/??AO??e?c??Y???&S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??~j?t??Q?|a??Ad;?O????Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????	???Aŏ1w-!??Y?#??????*effff^?@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???QI??!N܁oȢ@@)???????1????̏>@:Preprocessing2F
Iterator::Modelt$???~??!n{?a??B@)?*??	??1"?#2?"7@:Preprocessing2U
Iterator::Model::ParallelMapV2?<,Ԛ???!q?!??,@)?<,Ԛ???1q?!??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???~?:??!??~?==O@)9EGr???1ɾ=??(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Ƭ?!A?e	?@)??Ƭ?1A?e	?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???9#J??!??y;C*@) ?o_Ω?1??V?m?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?5?;N???!&qZ?%1@)???{????1??WV%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?I+???!Y?7?"?@)?I+???1Y?7?"?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??%?J@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???Q?`??Paa?n???	???!Ǻ?????	!       "	!       *	!       2$	?*=??K??? ?g?$??ŏ1w-!??!O??e?c??:	!       B	!       J$	?}??+????[A???2U0*???!???&S??R	!       Z$	?}??+????[A???2U0*???!???&S??JCPU_ONLYY??%?J@b Y      Y@q'?M?w,U@"?
both?Your program is POTENTIALLY input-bound because 48.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.6948% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 