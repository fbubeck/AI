?($	xJ????!??R>???ܵ?|???!sh??|???$	?{cQ@>pi޻?@ ??I@!??'??)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??&S??J{?/L???AU???N@??Y??ܥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~j?t???T㥛? ??A?j+?????Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q???????#?????Aj?t???Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Y??ڊ??_?L???A?sF????Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%??NbX9???A?????B??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_???MbX9??AB?f??j??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f???=,Ԛ???A??C?l??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+?????(\???A?? ?rh??Y??\m????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A????? ?rh??Av??????Y?N@aã?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	????K7??<?R?!???AL?
F%u??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?=?U?????%䃞??A&䃞ͪ??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????-???1??A?Zd;???Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^?I??xz?,C??A???H.??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&sh??|???ףp=
???A???Mb??Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f?????&?W??A?D?????Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o?ŏ1??????K7??A?QI??&??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?ZB??g??j+???Aj?q?????Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM??㥛? ???A???(\???Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~???NbX9???A???x?&??Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W[?????ı.n???Avq?-??Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ܵ?|???????Mb??A?Fx$??Y_?Qڛ?*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??g??s??!???(6A@)e?`TR'??1??/?R??@:Preprocessing2F
Iterator::Model?_vO??!2?z??A@)?J?4??1???O?5@:Preprocessing2U
Iterator::Model::ParallelMapV2K?46??!:?i?=?+@)K?46??1:?i?=?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?T???N??!g?®?P@)46<?R??1*Kݠ?&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?46<??!?Z?.??,@)??H?}??13??d@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceB>?٬???!j???!?@)B>?٬???1j???!?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??^)??!?J%???2@)?V-??1?t?~??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Dio??ɔ?!?3??@)Dio??ɔ?1?3??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no99?e2}@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	\??X?????0?N??????Mb??!ףp=
???	!       "	!       *	!       2$	??Z,?-???????Z???Fx$??!?D?????:	!       B	!       J$	???sv????bjp?????Q???!j?t???R	!       Z$	???sv????bjp?????Q???!j?t???JCPU_ONLYY9?e2}@b Y      Y@qۧ՘v?@@"?
both?Your program is POTENTIALLY input-bound because 52.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?33.4255% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 