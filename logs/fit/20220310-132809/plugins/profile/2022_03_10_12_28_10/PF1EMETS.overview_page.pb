?&$	$iA2'???3???ǿ??%r?|??!?I??І??$	?25??@Vsħ????Z????@!.??ńq@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?I??І????_?|???A?1"QhY??Y?i?*???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0&????}??????W??A??i??_??Y:!t?%??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0e??#?? ?#G:??A?S?D?[??Y???????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?R??F;???~?^???Aڌ?U???YM???X??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0#?	?????a?????A9}=_?\??Y??L?Nϛ?rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D?M?????đ??A?^?D???YsJ_9??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?!U?2??????0???ACV?zN??Y????'+??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?)s?????????J#??A???!o??Y?{?????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
w?}9?????jׄ???A???E???Y??yS?
??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?5??D?????n,(??A!?'?>??Y?'??&2??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Q??Z???P?Y??/??A??25	???Y~?[?~l??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????M?M?g??A?W?2??Yt???z???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o?j????pY?? ??A (??{???Y?@?)V??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??P????i????A$?@??Y?Դ?i???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?%r?|?????5[??A]??'???Y80?Qd???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0!??*?C???j+?????A??????Y?L??Ӏ??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?C5%Y?????q?????A???-??Y?3?z??rtrain 97*	-??憎@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat6?ڋh;??!??n??JB@)-]?6????1?,f?v?@@:Preprocessing2E
Iterator::Root7T??7???!H?????@)ޓ??Z???1?|m?? 0@:Preprocessing2T
Iterator::Root::ParallelMapV2??ao??!ޖ{T4C/@)??ao??1ޖ{T4C/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?E?x???!?-U[Q@)V-???1M-??p(@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?k?}?
??!?i???5 @)?k?}?
??1?i???5 @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateM???????!?(? 0@)??O?ް?1i?&? @:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???C6???!v?`A?3@)?ky?zۜ?1tġ?r@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?p?Qe??!?????@)?p?Qe??1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o1?;o@IuV"V?$X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?{?^????E)??|???j+?????!??_?|???	!       "	!       *	!       2$	???????],z@z??9}=_?\??!?S?D?[??:	!       B	!       J$	?6????Jb???3?z??!?i?*???R	!       Z$	?6????Jb???3?z??!?i?*???b	!       JCPU_ONLYYo1?;o@b quV"V?$X@Y      Y@q?,??mM@"?	
both?Your program is POTENTIALLY input-bound because 44.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?58.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 