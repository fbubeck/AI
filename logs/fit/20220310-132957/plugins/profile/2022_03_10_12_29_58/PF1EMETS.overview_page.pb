?&$	B	?GK	??j?Sm?????|	??!^0???~??$	?R?Ѱ$@??q????G??@!?????@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0:??????mR?X???A c?ZB>??Y??f?b??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0h\8?????Nw?x??A?D?[????YD?R?Z??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?]?pX???pt?????AM??E??Yo?j{??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0I?\߇?????ϝ`??A?W??I??Y??>V?ې?rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r??V??_??x?Z??A?u?;O<??YV,~SX???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???|	???m??E???A?h o???Y?9?S?ɒ?rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z?rL????H??Q,??Af??\???Y,?)???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?*??] ???.?u?;??Avp?71$??Y?dȱ???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
??r?????̳?V|C??A?M?????Yg?lt?O??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0^0???~???H?F?q??A?|$%=??Y?&????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0N*kg???K?Ƽ???At34???Y????=z??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails00e???.??((E+???A?[<?????Y???????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??L??S??S?h?w??A?z?"0???Y????W??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ɧǶ???m 6 B\??A{?Fw;??Y???~???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???d??????????A]???????YuZ?A????rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0xC8???
?2?&??AJ?E???Y?};????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????v??1ҋ??*??Af???i??Y?B?ʠڐ?rtrain 97*	p=
ף??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatB|`????!?yo?o?B@)g???d??1G???IdA@:Preprocessing2T
Iterator::Root::ParallelMapV2??O@???!?Ak^mK/@)??O@???1?Ak^mK/@:Preprocessing2E
Iterator::Root????????!?(w?0?@)?)?????1c????/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?!?uq??!??5b?3Q@)?25	ސ??1W?ґ?#$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceR*?	????!??_~?#@)R*?	????1??_~?#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenatejO?9????!?e'{1@)?t?i???1f?k??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap1zn?+??!nw,5@)??N$?j??1?{???@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?0????!?i??cB@)?0????1?i??cB@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-?+??
@I????'(X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?????????????????????!??Nw?x??	!       "	!       *	!       2$	???Y?????t???;???h o???!?|$%=??:	!       B	!       J$	?\??b???Xy?[?V,~SX???!uZ?A????R	!       Z$	?\??b???Xy?[?V,~SX???!uZ?A????b	!       JCPU_ONLYY-?+??
@b q????'(X@Y      Y@q?2Z??sP@"?	
both?Your program is POTENTIALLY input-bound because 46.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?65.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 