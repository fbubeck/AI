?($	h??_???0?!K?????ܵ?|??!0L?
F%??$	d?%J?@??C??@?Y??]@!Tg?x?P5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ׁsF???@?߾???A?|?5^???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z???c]?F??As??A??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~j?t???J{?/L???A?o_???Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??o_?????S????A0?'???Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S?!?uq??????Q??A]?Fx??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F???|?5^???Aq???h ??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o???T???????(??A?	?c??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W????ʡE???A??g??s??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??v??????A?S㥛???Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	F%u???z6?>W??ANbX9???Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
bX9?????@??ǘ??A?Q?|??Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0L?
F%???X????A?c]?F??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??g?????/?'??A??ݓ????Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F???g??s???A???S???Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??JY?8??,Ԛ????A??+e???Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY????ͪ??V??A"lxz?,??Y?-?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Q????*??	??A??ڊ?e??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8?????n?????A?q??????Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'?W???!?lV}??A+????Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZӼ????;Nё\??A?_vO??YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?|????H.?!??A?;Nё\??Y9??v????*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatHP?s??!????fB@)????Q??1jS?酺@@:Preprocessing2F
Iterator::Model??ʡE???!????[?B@)U0*????1?u??6@:Preprocessing2U
Iterator::Model::ParallelMapV2?Pk?w???!A??fa5-@)?Pk?w???1A??fa5-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?X?? ??!'?dO@)?~j?t???1??B&?H#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??e??a??!?<?/?G@)??e??a??1?<?/?G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate      ??!c???H)@)?46<??1?1?s??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapޓ??Z???!?Q??W0@)vOjM??1???K@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??镲??!??|???
@)??镲??1??|???
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9"?,1@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	l??=?i???K?~?????H.?!??!?@??ǘ??	!       "	!       *	!       2$	O?Ŗv??j???????;Nё\??!?c]?F??:	!       B	!       J$	,<?!e3??  %A?Ȉ?w-!?l??!U???N@??R	!       Z$	,<?!e3??  %A?Ȉ?w-!?l??!U???N@??JCPU_ONLYY"?,1@b Y      Y@q????bT@"?
both?Your program is POTENTIALLY input-bound because 54.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?81.5325% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 