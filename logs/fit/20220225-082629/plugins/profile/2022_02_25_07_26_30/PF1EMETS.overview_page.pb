?&$	 .t???1Hv???p?4(???!?V횐??$	??z?V?@??j????5????@!?3????@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???x????	?L?n??A???!b??Yj?0
???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??E}?;???im?k??A6??????Y??:U?g??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??7?-:??aE|??AD?.l?V??YM??u??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails06=((E+??Lݕ]0???A?^????Y?j??Ք?rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails08i̓????????AV??????Y????y??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?$"?? ??fI??Z???A??<????Y??!??T??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??$W?????`ũ???A"?4????Y?ECƣT??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	a8?0C?????J#f???A?D??Ӝ??Y^???????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
Z5??#??ȔA????A?D?+??Y????}??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0p?4(????|[?T??A}[?T???Y2??8*7??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0l??7????U?3??A?#?&ݖ??Y?u6????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$	?P(??9d?b???A???C?X??Y???D??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0'?ei?????Ͻ???A?H?"i7??Y?|?q??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??eOB????7i??A???????Y???=^H??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?V횐????jGq??A??[z??Y??rg&??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??8ӄm??$_	??.??A???ۂ???Y(??쿦?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??dV?0???!? ?&??A?0???C??Y?qn?rtrain 97*	???Sㅊ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?!8.????!?
? ?C@)?ȳ???1?XMA@:Preprocessing2T
Iterator::Root::ParallelMapV2?fG?????!?b?FF/@)?fG?????1?b?FF/@:Preprocessing2E
Iterator::RootJ??	?y??!i????T>@)??8ӄ???1'?5?c-@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipF
e?????!&݀??jQ@)?d???1?????'@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice#??~j???!?48S#@)#??~j???1?48S#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatea?xwd???![?????.@)@??߼8??1?lc?V7@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??q?߅??!?j????3@)????e??1? ?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???a????!u??lV@)???a????1u??lV@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????t@IbBXX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	w?(????ik?i???aE|??!??jGq??	!       "	!       *	!       2$	???n????xe? L???#?&ݖ??!???!b??:	!       B	!       J$	???[?|?????*????????y??!???=^H??R	!       Z$	???[?|?????*????????y??!???=^H??b	!       JCPU_ONLYY????t@b qbBXX@Y      Y@qYWˌ?$I@"?	
both?Your program is POTENTIALLY input-bound because 48.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?50.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 