?($	zGc????:x3????Y?? ???!f?c]?F??$	?0m??@?1???@?b??	@!??L?J	6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?V?/?'???O??e??A??/?$??YV-?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???{????w??/???A??(???Y$????ۧ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o?????@??ǘ??A?-?????Y??\m????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ???S??:??A?	?c??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??JY?8???V-??A???QI???Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??H.???"lxz?,??A,e?X??Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?2ı.n?????<,???A??7??d??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Mb??T㥛? ??A?:M???Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&lxz?,C??B`??"???A?/?'??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	xz?,C???C??????Aj?q?????Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??^)??d?]K???A@?߾???Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F???JY?8???A?Zd;??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& ?o_????JY?8???A???????Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V-??O@a????A??a??4??Y_?L???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q?-???O@a????A!?lV}??Y????z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R??l	??g???A??6???Y o?ŏ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??QI???????H.??A&S????YM?J???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s????m???????As??A??Y????z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&g??j+????C??????A?q??????Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ꕲq???H?}8g??A??Q???Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ???j?q?????A"?uq??Ym???{???*	43333_?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK?46??!????@@)'???????1?v	)-?@:Preprocessing2F
Iterator::ModelTR'?????!H/rQ?A@)U???N@??1????.7@:Preprocessing2U
Iterator::Model::ParallelMapV2?!??u???!R?k??)@)?!??u???1R?k??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?!??u???!\??F?P@)??s????1X[??'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?	?c??!ծD?J?*@)?H?}8??1?????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice<?R?!???!"?m?	?@)<?R?!???1"?m?	?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??v????!4>?H?2@)???h o??1?ro???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Q?|a??!<G??%@)Q?|a??1<G??%@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Gc֐e@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	w{??)????\??M??j?q?????!?JY?8???	!       "	!       *	!       2$	.?????,}?%???"?uq??!?Zd;??:	!       B	!       J$	??7?U????coՕ??(??0??!_?L???R	!       Z$	??7?U????coՕ??(??0??!_?L???JCPU_ONLYYGc֐e@b Y      Y@qH??
?=@"?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?29.6681% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 