?($	C??"C6????t?F???l	??g???!ݵ?|?3@$	?R
?~@?$S??@?g??????!?0?U??*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?x?&1????H.?!??ADio?????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??k	?????<,Ԛ???A?HP???YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|?3@a2U0*???A?
F%u??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p?????&S??A?JY?8???Y?^)?Ǫ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?v??/??a2U0*???A?&S???Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
????H.?!??Af??a????Y+??Χ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n???]m???{??A???H??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:??jM??St??Affffff??Y??b?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&I.?!?????+e?X??At??????YB`??"۩?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?V-??0?'???A'1?Z??Y9??m4???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
EGr?????A`??"??As??A???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&g??j+????6?[ ??A?%䃞???Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?k	??g????????A?	?c??Y??ܥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????^?I+??A?D?????Y?|гY???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O??-!?lV??Ae?X???Y?2ı.n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾???D????9??A?D???J??Y<?R?!???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????M???c?ZB??Alxz?,C??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????]?C?????A???QI??Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??x?&1???Pk?w???AX9??v???Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?j+??????q??????Aۊ?e????Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&l	??g?????H.?!??A?o_???Yn????*dffff??@)      P=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatt$???~??!6ہ??B@)|a2U0??1???#?@@:Preprocessing2F
Iterator::Model????߾??!G"?t?SA@)?k	??g??1d4@:Preprocessing2U
Iterator::Model::ParallelMapV2????<,??!?~??&-@)????<,??1?~??&-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk+??ݓ??!????/VP@)Ș?????1[pE ?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?G?z??!?k?+?@)?G?z??1?k?+?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?1w-!??!??|??-@)??y?)??1E?Z?0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?A`??"??!W ???3@)??:M??1=??V@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ZӼ???!?lM?@)??ZӼ???1?lM?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9xl??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	b????????@}????H.?!??!a2U0*???	!       "	!       *	!       2$	?'g????SѼhsC???o_???!ffffff??:	!       B	!       J$	,C??6??$_?Z??h??|?5??!??b?=??R	!       Z$	,C??6??$_?Z??h??|?5??!??b?=??JCPU_ONLYYxl??@b Y      Y@q????U(>@"?
both?Your program is POTENTIALLY input-bound because 57.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.1576% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 