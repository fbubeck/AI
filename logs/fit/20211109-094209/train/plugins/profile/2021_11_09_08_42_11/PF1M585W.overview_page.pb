?($	?Ξ????aN??l???&S??:??!a??+e??$	?&?%p?@;{??Nz@?V3?@!?d???}6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????????????A䃞ͪ???YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a???0?*??A??HP??Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9????D????AL?
F%u??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY???????߾??AԚ?????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~??k????x?&1??A|a2U0*??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F??????A?\m?????Y?/?'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a??+e??[Ӽ???AQ?|a2??Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ???L?
F%u??A`vOj??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???9#J?????S???A?t?V??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??镲?????QI???A$???????Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
c?=yX??+????A8gDio??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}???|a2U??A?Zd;??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?o_???aTR'????A-???????Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e????U??????A(~??k	??Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[???l	??g???A???<,???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V???8EGr???A*:??H??YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????$(~??k??A?߾?3??YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?m4??@????H.???A????K7??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w????jM??St??A?|?5^???Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w????n4??@???A?O??n??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S??:????HP??A*:??H??Yn????*	?????ȋ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatn????!?uȊĢA@)w-!?l??1?????0@@:Preprocessing2F
Iterator::Model??3????!ϑ(?PA@)*??D???1?????5@:Preprocessing2U
Iterator::Model::ParallelMapV2??H.?!??!^Q??R?)@)??H.?!??1^Q??R?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)??0???!?ky?WP@)|a2U0*??1???N?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??镲??!????@)??镲??1????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate"?uq??!y.?.S5,@)%u???1߯,?&t@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=?U????!iɨ?
3@)????z??1? 8E?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*p_?Q??!?l??@)p_?Q??1?l??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no92?E??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	*jU??????~????HP??![Ӽ???	!       "	!       *	!       2$	?鯁ۂ???=?\??*:??H??!?\m?????:	!       B	!       J$	?8(J???q?;?W??%u???!P?s???R	!       Z$	?8(J???q?;?W??%u???!P?s???JCPU_ONLYY2?E??@b Y      Y@qZt?A\??@"?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.81% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 