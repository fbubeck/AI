?($	)??=I??惬k????Έ?????!.???1???$	+?n^n@????߄@_ߨڦ?@!ʡ?j?;@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???x?&??S??:??A??{??P??Y}?5^?I??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF??$???????A؁sF????Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??4?8E??j?t???A?0?*???Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S??????:M???A2U0*???Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H.?!?????ڊ?e??A?h o???Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Dio???????n????A??C?l??Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&.???1?????ڊ?e??Ak?w??#??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z?):?????A?f????A?:M???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ??R???Q??A5?8EGr??Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	)\???(?????<,???A=?U?????YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
9??m4??????镲??A?H.?!???Y??D????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{????_?L??A???????YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE???QI??&??A?ݓ??Z??Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W?/?'??"?uq??A8gDio??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ???(\?????AӼ????YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0*??W[??????A??HP??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??v????{?G?z??A??<,Ԛ??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??T?????9??m4???A?[ A?c??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ????*:??H??A?ׁsF???Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F???_?L??A??镲??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Έ??????f??j+??A?QI??&??Y???<,Ԛ?*	????̦?@2F
Iterator::Model??T?????!w?B?2gH@)?7??d???1?}C???B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?St$????!K%fh|;@)S?!?uq??1?Es9@:Preprocessing2U
Iterator::Model::ParallelMapV2??b?=??!?#?3??&@)??b?=??1?#?3??&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<?R?!???!?y?]͘I@)y?&1???1??D??"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate? ?	???!????F?$@)      ??1?<?){?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????!r?8	W@)????1r?8	W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???S???!?Ɇ???,@)?b?=y??1C???u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??+e???!4?1tI @)??+e???14?1tI @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|??!?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	????3??|?Ԑv????f??j+??!?A?f????	!       "	!       *	!       2$	ۙpW٥????*????QI??&??!k?w??#??:	!       B	!       J$	h5?}ť?U??l1+??46<?R??!}?5^?I??R	!       Z$	h5?}ť?U??l1+??46<?R??!}?5^?I??JCPU_ONLYY|??!?@b Y      Y@q??8?J?:@"?
both?Your program is POTENTIALLY input-bound because 55.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?26.7082% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 