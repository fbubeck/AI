?&$	?????Q?\M(??Xr????!?2T?Tz??$	b??-g?@SPbT?G??3?iR?@!??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0VE?ɨ2???ME*?-??A??9????Y?-?|????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?\6:?'??B??=???A??x@???Y???	/??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ʾ+?????g	2*??Ag{?????Y2Ƈ?˖?rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ȷw?R???p!????Aڏ?a??Y?b)????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???L???@x?=\??A???C???Y?T[r??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0;?5Y????EИI??A??@????YJ?????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?r.?Ue??e?fb??A???B??YöE?2??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	???<????{?E{??AND??~???Y[?Qf??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?2T?Tz??G?&jin??A?5[y????Y?˛õړ?rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?z??>??7Ou??p??A?,?????Y??d#ٓ?rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?S?K???*s??????A??)1	??Y?7??d???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o??g??ӅX????A2 Ǟ=??Y?vLݕ]??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Xr????NbX9???A%̴?+??Y??rJ@L??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?F?????4???????A?mr????Y???$????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ɏ?k???KO?\??A???????Y5?;???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0_????@?????0a??A??????Y??,????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??S? P??!XU/????A??n????Y??/?$??rtrain 97*	_?Iz?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??lu9??!;???$?B@)????k??1l?pKZA@:Preprocessing2T
Iterator::Root::ParallelMapV2 ??a??!?ʙ2x?0@) ??a??1?ʙ2x?0@:Preprocessing2E
Iterator::Root??aMeQ??!?????@@)%y???A??13_?4O}0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip4H?Sȕ??!?u7&??P@)?I~įX??1???"?B$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?r?4???!?Y 9??#@)?r?4???1?Y 9??#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatef1???6??!????-@)P??|zl??1?0??DQ@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap֭???7??!?C6?l3@)???g???1n??zO?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*%???7ی?!?????`@)%???7ی?1?????`@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?{IPQ?@I"?}u}X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?????5????ÂS??@x?=\??!B??=???	!       "	!       *	!       2$	??????????????x@???!?5[y????:	!       B	!       J$	yrM??Β??????d?[?Qf??!öE?2??R	!       Z$	yrM??Β??????d?[?Qf??!öE?2??b	!       JCPU_ONLYY?{IPQ?@b q"?}u}X@Y      Y@qax??T@"?	
both?Your program is POTENTIALLY input-bound because 47.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?83.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 