?&$	d?G??????-???$??=Զa??!u?yƾd??$	r??xn6@G9??UC???t???@!['????@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Q???? %vmo???A#h?$???Y,????`??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0u?yƾd??-y??ACW"P????Yt??%???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0l???,?????m3??ASͬ????Y?v?E??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?=$|?o??p?܁:??A?^Pj??Ya?????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?5w??\?????????A??F???Y??oa?x??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0{2???4????d9	???A?\?	???Y?% ??*??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????$@M-[??AL?e?%???Y{?%T??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?o{??v???????S??A? ?m?8??YS[? ???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
fN????????g\8??A?&??n??Y????1v??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?? ?> ??ByGsd??A?ؘ????YHG?ŧ??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????-?2????A??a0???Y?۽?'G??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails01?~?٭???	ܺ????A'"????Y?e?O7P??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0=Զa?????????A???%P??Y????c>??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0R?r????q?a????A??_ ??Y?в???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0BȗP???xρ???A]??k??Y㪲?????rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?t["???!???s??A??G6W???Y?an?r??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??jGq???6?e????A&s,????Y??????rtrain 97*	?x?&1(}@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatz?]?zk??!0??u!A@)<Mf?????1?dS?@:Preprocessing2T
Iterator::Root::ParallelMapV2G?@?]???!??42@)G?@?]???1??42@:Preprocessing2E
Iterator::Root?m??)??!??J??A@)??J???10??#$<1@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice4Lm?????!ܩ֓?	#@)4Lm?????1ܩ֓?	#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?\kF??!2?Z??#P@)OGɫs??1T?x?!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate"rl=C??!q>??o?0@)?v/?ɡ?1???Z?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapK?P???!??)??5@)??}????15?F?U@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?t_?l??!I??O?@)?t_?l??1I??O?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9y???@IdI?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	zHE?????0?5????-?2????!-y??	!       "	!       *	!       2$	?iɖ???j?e???????%P??!?\?	???:	!       B	!       J$	??N??G???[M]??an?r??!??oa?x??R	!       Z$	??N??G???[M]??an?r??!??oa?x??b	!       JCPU_ONLYYy???@b qdI?X@Y      Y@qVn|O?S@"?	
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?76.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 