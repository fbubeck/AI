?($	C???=/???6
????????&S??!??ͪ????$	է*.@̿??P1@??I)A@!?Y<??,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$n????*:??H??A:#J{?/??Y??K7?A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?`TR'??Dio?????A?rh??|??Y?"??~j??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ͪ?????镲q??As??A??Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Z??ڊ???????x???AEGr????Y?5?;Nѡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?<,Ԛ???+????A6?>W[???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u?????|?5^??A?/?'??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$????V-???A?5?;N???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??A?f??#J{?/L??A㥛? ???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ŏ1w???W?2??A??ǘ????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	w-!?l???Fx$??AQ?|a??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
+??	h???8??m4??Aa??+e??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'??????e?c]???A㥛? ???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????s??A??A	??g????YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h"lxz???-C??6??A	??g????Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?7??d????#??????A$???????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????T???N??A??&???YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?镲q??s??A???AT㥛? ??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ܵ?|????uq???A?-?????Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM????o_??A?;Nё\??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\?????:M??A?Ǻ????Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S????(\?¥?A???߾??YV-???*???????@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatH?z?G??!?+??)A@)Ǻ????1=??C??@:Preprocessing2F
Iterator::Model??镲??!??u?@@)L?
F%u??1G?zMiT2@:Preprocessing2U
Iterator::Model::ParallelMapV2*:??H??!?a>?/@)*:??H??1?a>?/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipJ{?/L???!?yyňP@)f??a?ִ?1????$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?t?V??!kĄ07 .@)K?=?U??1g??	@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice-!?lV??!n?(Ue"@)-!?lV??1n?(Ue"@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??%䃞??!???,x5@) ?o_Ω?1?m?C?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*n????!h?q?m?@)n????1h?q?m?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9#NK,o@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?????????	?????(\?¥?!????x???	!       "	!       *	!       2$	?Ǻ????'?'???????߾??!s??A??:	!       B	!       J$	\??0p?????S?????ZӼ???!??K7?A??R	!       Z$	\??0p?????S?????ZӼ???!??K7?A??JCPU_ONLYY#NK,o@b Y      Y@qR?9??>@"?
both?Your program is POTENTIALLY input-bound because 53.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.6082% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 