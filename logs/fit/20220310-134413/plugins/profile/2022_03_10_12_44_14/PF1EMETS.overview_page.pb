?&$	??!M`????Rd^T?? ??Udt??!?+ٱ???$	?Ȣ?*?@&?? ????.?	@!??L???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0x?~?~???1?:9Cq??A?*4?f??Y? x|{א?rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails07??­??)z?c????A9毐???Y?`????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0P???????}A	]??A$?`S?Q??Y???ْU??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????????c??A???$????Y?R?!?u??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???
???V?zNz???A2q? ???Y?&?????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C??6??82??????Aȳ˷>??Y???쟧??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0u=?u???K?H??r??A?? ?rh??YX??C???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??噗?????H?,??A*V?????Y?mP?????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
 ??Udt????a??4??A$	?P???Y"q??]??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??,z???|?ڥ???A?jdWZF??Y??[????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,g*??'i?????A?Xl?????Y?rJ@L?rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r??9???T???r???A?	1?Tm??YHP?s??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0? "5?b???:U?g$??AԵ?>U???Y???g????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0|a2U0*???-?????A??s?????Y??K?[???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?+ٱ???,??ص???A?w?Go???Y?i? ?Ӛ?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0>yX?5????T?????AZ)r?#??YY???-??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0_A??h:?????????A2?m??f??YX?????rtrain 97*	9??v???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatQ??????!?ڑ??A@)y?\??1-҆?a_@@:Preprocessing2E
Iterator::Root??#bJ$??!?????@@)?ȑ??ȷ?1??<ij1@:Preprocessing2T
Iterator::Root::ParallelMapV2"??????!X???y0@)"??????1X???y0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipL?K?1???!????P@)Z??M??1?&=bS0&@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicer4GV~??!P6?W?$@)r4GV~??1P6?W?$@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?׺????!?@#N??.@){?\?&???1T(??R@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???0??!c????3@)??-Θ?1???{?)@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*T?D?[ʉ?!?^>?}?@)T?D?[ʉ?1?^>?}?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9y???@I7???
X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??$Yِ???????-?????!)z?c????	!       "	!       *	!       2$	I:[????4D????*4?f??!?w?Go???:	!       B	!       J$	?3?0O???G?0D??i???K?[???!?i? ?Ӛ?R	!       Z$	?3?0O???G?0D??i???K?[???!?i? ?Ӛ?b	!       JCPU_ONLYYy???@b q7???
X@Y      Y@qB?}??U@"?	
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?86.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 