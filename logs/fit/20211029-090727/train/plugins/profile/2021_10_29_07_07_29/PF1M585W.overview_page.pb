?($	 W?,"??&F?-Z????? ???!
ףp=?@$	?????%@???$o?@s?#?X_@!Nct??)0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$㥛? ???[????<??A??z6???Yףp=
׳?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=?@???S???A?J?4??Y????ҿ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??@7?[ A??AǺ?????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????<,??ˡE?????A_?Q???Y?c]?F??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R????#?????A@a??+??YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?#???????N@a???A???????Y?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&㥛? ?????N@a??A?.n????Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5^?I????#?????A?ܵ?|???Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'??????o??A"lxz?,??Y???V?/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???(\???'?W???AjM??St??Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
O??e?c??? ?	???ADio?????Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio???lV}???A??+e???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"lxz?,??%??C???A[B>?٬??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S?!?uq??h??s???A1?*????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ???????0?*??A?Zd;???YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&aTR'?????>W[????A=?U????Y$????ۧ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~??k	??????QI??AY?? ???Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????$(~??k??A??ׁsF??YNё\?C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=
ףp=??X9??v???A{?/L?
??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]??7?A`????AǺ????YX9??v??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ???(??y??A???JY???Y??D????*	gffff?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK?46??!??|A@)??\m????1?R?????@:Preprocessing2F
Iterator::Model9??m4???!r0KB@)??^??1???o??7@:Preprocessing2U
Iterator::Model::ParallelMapV2?$??C??!?2!錓)@)?$??C??1?2!錓)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??A?f??!??Yc?$@)??A?f??1??Yc?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.???1???!????ϴO@)???S????1?V?4?n"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?|a2U??!??ä.@)?I+???1.?k?."@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??|гY??!??A?`4@)n????1?)???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??~j?t??!???+n@@)??~j?t??1???+n@@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9N? Н@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???????w?5q??(??y??!7?[ A??	!       "	!       *	!       2$	j????~???\h$?@?????JY???!?J?4??:	!       B	!       J$	?@?>?ȭ??m?an??vOjM??!????ҿ?R	!       Z$	?@?>?ȭ??m?an??vOjM??!????ҿ?JCPU_ONLYYN? Н@b Y      Y@qk?c9?3@"?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?19.5087% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 