?($	%?$?)??@q?N\Z???l??????!?? ?rh??$	?	4ak@???t?+@\???[@!?V???2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?{??Pk??:#J{?/??A????B???YbX9?ȶ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}???"??~j??A
ףp=
??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U0*?????\?C????A??Q???Y?V-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4????s????A1?Zd??YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZӼ????鷯??A?C??????Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??JY?8???JY?8???A?H.?!???Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0*???W?2ı??A???x?&??Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&????t?V??AF%u???Y?????K??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ڊ?e????b?=y??Alxz?,C??Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	/n?????=?U???A?s?????Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
xz?,C???5^?I??A??s????Y???K7???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX?5??8??d?`??AjM????Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`????}8gD??A?D?????Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?	???c?ZB??A?=yX???Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"??u????jM??St??A?ŏ1w??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?JY?8????:pΈ??A?s?????Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&鷯????7??d???A??Q????Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q?????????(??A????K7??Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?rh????|гY??A?b?=y??YNё\?C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C??????*:??H??A1?Zd??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?l??????]?Fx??A?lV}???Y#??~j???*	fffff~?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!??RR??A@)?m4??@??1XÌC?p@@:Preprocessing2F
Iterator::Model?!??u???!?=?-B@)?,C????1G?$-F?7@:Preprocessing2U
Iterator::Model::ParallelMapV2??s????!R???')@)??s????1R???')@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Q?|??!?|???O@)~8gDi??1???u?#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??x?&1??!?2JUL@)??x?&1??1?2JUL@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??|гY??!?'[E?*@)n4??@???1­5.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??(???!??PX??2@)?? ?rh??1??6?F?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*O??e?c??!0*b?@g@)O??e?c??10*b?@g@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|S+??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	r?M2???X]S?݈??]?Fx??!??|гY??	!       "	!       *	!       2$	?T??????8???;????lV}???!1?Zd??:	!       B	!       J$	??sB}???u`??J?????ZӼ???!bX9?ȶ?R	!       Z$	??sB}???u`??J?????ZӼ???!bX9?ȶ?JCPU_ONLYY|S+??@b Y      Y@q舎Uš;@"?
both?Your program is POTENTIALLY input-bound because 56.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?27.6319% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 