?($	w??}~???v??J???
F%u??!j?t???$	?????@?c?@B???Z?@!?ܵ?z5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????????J+???A?W[?????Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG???+e?X??Aj?t???Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F???H.?!???AS??:??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.???T???N??A-??????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???Zd;?O???Aw??/???Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??<,Ԛ??????????AԚ?????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& c?ZB>??Ǻ????A?;Nё\??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D?l??????u?????A?[ A?c??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S????gDio????A.?!??u??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	????9#??;?O??n??Alxz?,C??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
z6?>W[???B?i?q??A??H.???Y?V-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t?????ܵ??Avq?-??Y??ʡE??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??o_??gDio????A@?߾???Y??ڊ?e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A???;Nё\??A??d?`T??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???+????A$(~??k??Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?q?????ꕲq???A?k	??g??Y?=yX???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)??vOjM??A????x???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?#??????%u???AjM??S??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?QI??&????V?/???A?G?z??Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B>?٬????U??????A;pΈ????Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?
F%u????(???A?C?l????Y???Mb??*	     ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?V-??!c?J\̖B@)?????15P??jA@:Preprocessing2F
Iterator::Model)\???(??!e???@D@)?w??#???1?J\?V	<@:Preprocessing2U
Iterator::Model::ParallelMapV2:??H???!(|????(@):??H???1(|????(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX9??v???!??c]s?M@)D????9??1R??L? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??6???!,Z~C@)??6???1,Z~C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatec?ZB>???!ọ'|?%@)???Mb??1?_???+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?? ???!?'|???,@)???~?:??1?????5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*w-!?l??!?b?J\?@)w-!?l??1?b?J\?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9nt?ԏ?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?tE?? ??p?>{Q???J+???!??ܵ??	!       "	!       *	!       2$	?p.??????ڐ
lߵ??C?l????!jM??S??:	!       B	!       J$	?y?T?Ӡ???)xְ???St$????!'???????R	!       Z$	?y?T?Ӡ???)xְ???St$????!'???????JCPU_ONLYYnt?ԏ?@b Y      Y@qnbE?U@"?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.3682% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 