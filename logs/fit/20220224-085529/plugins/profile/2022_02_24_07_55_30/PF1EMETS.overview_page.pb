?&$	G??Sg?????W;G??/??.??!?c"?ټ??$		????	@??j????&Ȋb?@!5?
?.?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0İØ????dw????A5F??j???Y"??????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0|a2U???Y?+?????AOu??p??Y?+I?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0mp"??u???ND???A??QI????Y???>Ȳ??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0/??.??b??????A??Z	?%??Y?+.??M??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????Q???rk?m?\??A?p>????Yc??????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ѓ2)??? ????As/0+???Y?Z?a/??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0b??U?????Z&????Ac??V???Yep??:ǐ?rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??ګ??????????A ?H? ??Y??????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
G9?M?a?????;???A
?F???Y?v/??Q??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0\[%X??H0?[w??A?'v?U??Y???@?w??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0AG?Z?Q????,????A6??Ң>??Y???'??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?.??[<??E?????AQ?|a2??Y;?/K;5??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???-I???J??????Aa?ri????Y"?4???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?c"?ټ??a?>#??A2 {?????Y?T???B??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0>?ɋL@??W@??>??AF$a?N??Y{?\?&??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?3??????ǡ~?f??A??
~b??Y???s????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???/f??????4c??A?????Y????Ɋ??rtrain 97*	;?O????@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?0??B???!?@??_?B@)>Ab?{???1???L??@@:Preprocessing2T
Iterator::Root::ParallelMapV2|???s??!Y֎?#0@)|???s??1Y֎?#0@:Preprocessing2E
Iterator::RootmU???!?#?x??@)???)???1?,t?/@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?q4GV~??!?N??N(@)?q4GV~??1?N??N(@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??\5???!?;?!Q@)???d????1??HH?N%@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??m?R]??!?hW???0@)???{??1??qd?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???A{???!??:Gj?4@)?'G?`??1ت7N?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*1}?!8.??!GG?!?f
@)1}?!8.??1GG?!?f
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9+I??(?	@I????2X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	1??????? ?p????????!???;???	!       "	!       *	!       2$	!?̢???N	?k??????Z	?%??! ?H? ??:	!       B	!       J$	G5??6ؑ????>??a?"?4???!;?/K;5??R	!       Z$	G5??6ؑ????>??a?"?4???!;?/K;5??b	!       JCPU_ONLYY+I??(?	@b q????2X@Y      Y@q?,Bz+R@"?	
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?72.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 