?($	??#???H????????JY?8??!Qk?w????$	???l@EV[w#@?
k?48@!?͎Z??-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q????M?J???Aё\?C???YK?46??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w????r??????A????K7??Y????߮?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t?????<,Ԛ??A??JY?8??YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ݓ????EGr????AǺ?????Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?}8g???%䃞???A?镲q??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):????W[??????A??b?=??Y?1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e?c]??f?c]?F??A	??g????Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s??????x???A??QI????Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?V???ܵ?|???A?(\?????Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	M?O???8gDio??Az?):????Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
????Mb???i?q????A?A`??"??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?'??Q?|a2??A?c?]K???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?'????T㥛? ??A%u???Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m?????QI???A?O??n??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????lxz?,C??A?m4??@??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2???ZB>????AQ?|a??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:??Y?8??m??A??Pk?w??Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&???q?-???A??e??a??Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3????.???1???A|a2U0??Y?lV}???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o???????߾??A???&S??Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??JY?8???&1???A??h o???Y
ףp=
??*	?????X?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???{????!<J0ڱ@@)d;?O????1?7?Vw=@:Preprocessing2F
Iterator::ModelV-????!?ԉ_??B@)?U??????1]D???6@:Preprocessing2U
Iterator::Model::ParallelMapV2ڬ?\mž?!???ꅣ-@)ڬ?\mž?1???ꅣ-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??1??%??!+v?O@)f??a?ִ?1blJVN$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?[ A???!??_5#@)?[ A???1??_5#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;pΈ????!?M???0@)?@??ǘ??1/XʛȔ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapDio?????!??fa?4@)K?=?U??1?s?pe.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*46<?R??!?m?@)46<?R??1?m?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9}?úkh@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???4qD???" 	???M?J???!.???1???	!       "	!       *	!       2$	HǫE?$???!?G?????h o???!??JY?8??:	!       B	!       J$	q??W??:8@??m???o_???!????߮?R	!       Z$	q??W??:8@??m???o_???!????߮?JCPU_ONLYY}?úkh@b Y      Y@q????B@"?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.8833% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 