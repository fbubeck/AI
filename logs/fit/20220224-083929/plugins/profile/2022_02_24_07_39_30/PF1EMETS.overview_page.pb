?&$	θߨ??I
3?|"??ʌ??^???!??.\???$	1?s?]@?P?$???ЁJ[@!?'(Q^@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?!??T2????A{????A??@????Y?oD??k??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?G?????\?W zR??A??1?3/??Y?˸?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0a⏢?????? ??z??A?'????Y%?}?e???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0k??qQ-?????0????A@?ŊL??Y8????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ʌ??^???i??Q???A???2???YG6u??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??.\?????~?n??A??ң????Y?M???
??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??f?b??/?????A?=]ݱ???Y?I?p??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	*?-9(??z?(???A????????YXXp?????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
o?EE?N??eÚʢ???Aɰ?72???Yf/?N[??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0X9??????z?Fw??A??x??[??YǁW˝???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??۟???????V??A?!S>U??Y?1Xq????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??'?????H?<???A??~?????Y??;??J??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????UK:??l??AW??yr??Y?ZD?7??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??D????\?#?????A2Ƈ????Y???Y????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0^d~????Y?_"?:??A?k?6???Y???C?X??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?{?ڥ??????	???Aj1x????Y˟o????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0 ?3h???*??????AE?A???YoJy?????rtrain 97*	\???("?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatL?uT5??!?!???A@),?/o???1ܹ	.?@@:Preprocessing2T
Iterator::Root::ParallelMapV2?O@????!U$QZ?1@)?O@????1U$QZ?1@:Preprocessing2E
Iterator::Root?`U??N??!z?jJ?@@)Gq?::???1?????(0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???????!???I?"@)???????1???I?"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?-?\o??!øJ?Z?P@)l#?	???1Y?ߍm?!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?sb?c??!????2@)Di???1:Vn?.!@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapF^????!S??
6@)?I?_{??1C??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?q4GV~??!?z&??)@)?q4GV~??1?z&??)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????&@I? '??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	mo`?[??GE]????eÚʢ???!/?????	!       "	!       *	!       2$	K??????S????N???=]ݱ???!??ң????:	!       B	!       J$	???2m??yv????m????Y????!?ZD?7??R	!       Z$	???2m??yv????m????Y????!?ZD?7??b	!       JCPU_ONLYY????&@b q? '??X@Y      Y@q?????cU@"?	
both?Your program is POTENTIALLY input-bound because 45.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?85.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 