?($	?v???^??????ܛ??Ș?????!=
ףp=??$	z$??@G?\@???@!?.?(??-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?B?i?q?????S???AJ+???Y?2ı.n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???9#J??'?W???A/n????Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#J{?/L???Zd;???Af?c]?F??YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W??6?>W[???A0L?
F%??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L?
F%u??e?`TR'??A?:M???Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V-??c?=yX??Am???????Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/??|??Pk???Aq???h ??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX?????St$???Aj?t???Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~????ܵ???A???Q???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	RI??&???f?c]?F??A??_vO??Yj?q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?ʡE?????|гY???At??????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?i?q??????%䃞??A????o??Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞??????ׁs??A?\m?????Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=
ףp=????7??d??A?&?W??Y??C?l???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}??ۊ?e????A??y?)??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a??+e????n????A?@??ǘ??Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?=yX??q=
ףp??AO??e?c??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?????:M??A?HP???Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J+???????B???A???<,???Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a????z6???A]m???{??Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș?????Z??ڊ???AX?2ı.??YZd;?O???*	?????ː@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?}8gD??!????O]B@)??4?8E??1?A???@@:Preprocessing2F
Iterator::Model>yX?5???!?j?ڝLA@)u?V??1??5E5@:Preprocessing2U
Iterator::Model::ParallelMapV2_?L???!vy????*@)_?L???1vy????*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??"??~??!?J??YP@)5^?I??1Wk?.6?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??????!)?oo-@)??S㥛??1??Dp??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?1??%???!?%??@)?1??%???1?%??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?s????!Ξ???2@)bX9?Ȧ?1?\?/G?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?q??????!Qcn	?8@)?q??????1Qcn	?8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9H?,@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??3 ?-????$??p??Z??ڊ???!??7??d??	!       "	!       *	!       2$	]Z;?\??e?G???X?2ı.??!?&?W??:	!       B	!       J$	?,?>???5??????M??St$??!?2ı.n??R	!       Z$	?,?>???5??????M??St$??!?2ı.n??JCPU_ONLYYH?,@b Y      Y@q?????S@"?
both?Your program is POTENTIALLY input-bound because 53.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?79.2029% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 