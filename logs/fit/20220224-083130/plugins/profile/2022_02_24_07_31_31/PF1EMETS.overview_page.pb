?&$	?	g`?\??E?\?[:???\o?????!??T??7??$	?m`?@???9?
???U;@!?A??i@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?tB? ???????A<??~K??Y?&jin???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??~???????w???ApxADj???Y A?c?]??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?3??O???I? OZ??A?.??????YY??Z???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0.?ED1?????Pk?w??Ad???????Y???I???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?և?F-??h@?5_??A?,????Y?_!se??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????j??w?>XƆ??AKxB?????Y܁:?э??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??T??7??`??9z??Ab?G,??Y?=?#d??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??(???$bJ$????AU???????Yj??%??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
-B?4-??h?RD?U??A??0?*??Y?k???P??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0nYk(????ͮ{+??A??F??R??YUMu??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??i???????????A7l[?? ??YP4`?_??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????)???ۡa1???A??? !???Y	??YK??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????6??E?
)????A???????Y???Σ???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0T6??,
??!?> ?M??Ac?~?x???YT㥛? ??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?\o?????JV?????AY??+????Y???N??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0n??[???릔?J??A???x???Y?ǵ?b???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??'?H???^????A[Υ?????Y????i??rtrain 97*	/?$??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??????!!?'/C?A@)6rݔ?Z??1??G,?@@:Preprocessing2T
Iterator::Root::ParallelMapV2??ܵ?!???:?l/@)??ܵ?1???:?l/@:Preprocessing2E
Iterator::Root?fH???!??=?J?@)80?Qd???1??@?(/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip[x^*6???!???C-Q@)Bx?q?Z??1??_f a$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?^f?(??!?$N!?#@)?^f?(??1?$N!?#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate3d?????!Y? ???1@){3j?J>??1<??,?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap*q㊋??!Sy?0?6@)?W?B?_??1????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?}:3P??!kخwn1@)?}:3P??1kخwn1@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 42.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9_?N??@I???*9X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??阡??l1ek?~??릔?J??!?ۡa1???	!       "	!       *	!       2$	JfL$???K˴??g??Y??+????!b?G,??:	!       B	!       J$	!??8?B?????h?e?P4`?_??! A?c?]??R	!       Z$	!??8?B?????h?e?P4`?_??! A?c?]??b	!       JCPU_ONLYY_?N??@b q???*9X@Y      Y@q??{???R@"?	
both?Your program is POTENTIALLY input-bound because 42.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?75.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 