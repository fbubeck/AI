?&$	?}??2#???????i??%??1 ??!,~SX? ??$	???t?@@WQ???????Ư5<@!???|?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0b?A
????????K7??AE?a????Y2???A???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???,z???I??Gp??Akծ	i??Y*??Dؐ?rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0q<??f???:s	???A[rP???Y??d??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?v??-u???S????A?\?gA(??Y??4}v??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?|?F???????@-??A?Q??/??Y???v?
??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0mr??	????E?T???A????b)??Y̳?V|C??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0%??1 ??????:??AcD?в???Y?Y??U???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	n2????Y???"??A???f???Y@OI???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?????????GnM?-??AL⬈????Y5ӽN?˒?rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??*?b???J?E???AZEh????Y?Xm?_u??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?<?O?????2??A?i?L??Y?s????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0I0??Z????)s?????A?ِf??Yh?ej???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,~SX? ??S?u8?J??A??k]j???Y??W??"??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??3?????/L?
F??AL5????Yϻ??0(??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C???ΰ?????w??Amscz???YL?uT5??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?k$	???????l???A$???????YG6u??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z???-????3?Ib??A?D???V??Y?[z4??rtrain 97*	j?t??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?K?[????!cPY??C@)???zO???1yR.??@@:Preprocessing2T
Iterator::Root::ParallelMapV2r??9???!?z?U?0@)r??9???1?z?U?0@:Preprocessing2E
Iterator::Root?-v??2??!??~?<X?@)??3??X??1z?:?1.@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceS?o*Ra??!۽??!&@)S?o*Ra??1۽??!&@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip	6??g??!R???)Q@)?ǁW˭?1?ʳ? $@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???	?ԟ?!I??W9?@)???	?ԟ?1I??W9?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???k????!??s/??.@)zލ?A??1JW??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??L?????!³?~v?3@)&??i? ??1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????<-@I{?3?&X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	9??B??????-h??????:??!S?u8?J??	!       "	!       *	!       2$	cw?????*o-????[rP???!??k]j???:	!       B	!       J$	ۃ?/???# ?#b????v?
??!??W??"??R	!       Z$	ۃ?/???# ?#b????v?
??!??W??"??b	!       JCPU_ONLYY????<-@b q{?3?&X@Y      Y@q?@l?{T@"?	
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?81.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 